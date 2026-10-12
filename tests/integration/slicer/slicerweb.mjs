// Install the built wheel through SlicerWeb, then run the same MRML/widget tests as desktop.
import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import { cp, readFile, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

const { positionals, values } = parseArgs({ allowPositionals: true, options: {
  fixtures: { type: "string" }, "allow-software": { type: "boolean", default: false },
  "require-f16": { type: "boolean", default: false },
} });
assert.equal(positionals.length, 2, "Usage: slicerweb.mjs SOURCE OUTPUT [--fixtures DIR --allow-software]");
const [source, output] = positionals.map((value) => path.resolve(value));
const { chromium } = await import(pathToFileURL(path.join(source, "web/node_modules/playwright-core/index.mjs")));
const testCode = await readFile(new URL("module.py", import.meta.url), "utf8");
const info = JSON.parse(await readFile(path.join(source, "../wheels/build-info.json"), "utf8"));
const index = JSON.parse(await readFile(path.join(output, "wheels/index.json"), "utf8"));
assert.equal(index.extensions.length, 1);
assert.deepEqual(index.extensions[0].modules, ["VesselRaySegmentationDemo"]);
const runtimeTests = fileURLToPath(new URL("../runtime/", import.meta.url));
let runtimePackage;
let workerFiles;
if (values.fixtures) {
  const staging = path.join(output, "runtime-test");
  runtimePackage = JSON.parse(execFileSync("python3", [path.join(runtimeTests, "stage_browser.py"),
    path.join(output, "wheels", index.extensions[0].wheel), staging], { encoding: "utf8" }));
  const fixtures = path.resolve(values.fixtures);
  const cases = JSON.parse(await readFile(path.join(fixtures, "cases.json"), "utf8"));
  for (const name of ["cases.json", ...cases.map((spec) => spec.model)]) {
    await cp(path.join(fixtures, name), path.join(staging, name));
  }
  for (const name of ["browser.mjs", "model_o.mjs"]) {
    await cp(path.join(runtimeTests, name), path.join(staging, name));
  }
  await cp(path.join(fixtures, "model_o"), path.join(staging, "model_o"), { recursive: true });
  for (const directory of ["public", "dist/app"]) {
    await cp(staging, path.join(source, "web", directory, "runtime-test"), { recursive: true });
  }
  workerFiles = Object.fromEntries(await Promise.all(["application.py", "worker.py"].map(async (name) =>
    [`/work/${name}`, await readFile(path.join(runtimeTests, name), "utf8")])));
}
const server = spawn("npm", ["run", "preview", "--", "--host", "127.0.0.1", "--port", "4173", "--strictPort"],
  { cwd: path.join(source, "web"), stdio: "inherit", env: { ...process.env,
    SLICERWEB_WHEELS: path.resolve(source, "../wheels"),
    SLICERWEB_EXTENSIONS: path.join(output, "wheels"),
  } });
let browser;
await mkdir(path.join(output, "results"), { recursive: true });
try {
  const url = "http://127.0.0.1:4173/?sample=";
  for (let attempt = 0; ; attempt++) {
    if (server.exitCode !== null) throw new Error(`SlicerWeb preview exited: ${server.exitCode}`);
    try { if ((await fetch(url)).ok) break; } catch {}
    if (attempt >= 60) throw new Error("SlicerWeb preview server did not start");
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  browser = await chromium.launch({ channel: "chrome", headless: true,
    args: ["--ignore-gpu-blocklist",
      ...(!runtimePackage || values["allow-software"] ? ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] : []),
      ...(values["allow-software"] ? ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader"] : [])] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  if (runtimePackage) {
    await page.addInitScript((wheel) => {
      localStorage.setItem("slicerweb.extensions", JSON.stringify([
        new URL(`extensions/${wheel}`, location.href).href,
      ]));
    }, index.extensions[0].wheel);
  }
  page.setDefaultTimeout(300000);
  page.on("pageerror", (error) => console.error(String(error)));
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) console.log(`[${message.type()}] ${message.text()}`);
  });
  await page.goto(url);
  await page.waitForFunction(() => ["ready", "failed", "error"].includes(window.slicerWeb?.store?.status));
  assert.equal(await page.evaluate(() => window.slicerWeb.store.status), "ready",
    "SlicerWeb application startup failed");
  const report = await page.evaluate(async ({ wheel, code }) => {
    const runtime = window.slicerWeb;
    await runtime.installWheel(new URL(`extensions/${wheel}`, location.href).href);
    await runtime.loadExtensionPackages();
    await runtime.loadModulesWithPackages(true);
    runtime.pyodide.runPython(code);
    return JSON.parse(runtime.pyodide.runPython(`
import json, sysconfig
json.dumps(test_installed_module(sysconfig.get_paths()["purelib"] + "/slicer_home"))
`));
  }, { wheel: index.extensions[0].wheel, code: testCode });
  report.slicerweb_revision = info.slicerweb.commit;
  report.browser = browser.version();
  await page.evaluate(() => window.slicerWeb.pyodide.runPython('slicer.util.selectModule("VesselRaySegmentationDemo")'));
  await page.getByRole("button", { name: "Create demo curve", exact: true }).click();
  await page.waitForFunction(() => window.slicerWeb.pyodide.runPython(
    'slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLMarkupsCurveNode")') === 2);
  report.browser_button_click = "passed";
  if (runtimePackage) {
    const installedHashes = await page.evaluate(async (packaged) => {
      const pyodide = window.slicerWeb.pyodide;
      const root = pyodide.runPython('import sysconfig; sysconfig.get_paths()["purelib"]');
      const runtimeRoot = `${root}/${packaged.wheel_prefix}`;
      const installedManifest = JSON.parse(new TextDecoder().decode(
        pyodide.FS.readFile(`${runtimeRoot}manifest.json`)));
      if (JSON.stringify(installedManifest) !== JSON.stringify(packaged.manifest)) {
        throw new Error("Installed wheel and tested WASM manifests differ");
      }
      const hashes = {};
      for (const [name, expected] of Object.entries(installedManifest.files)) {
        const bytes = pyodide.FS.readFile(runtimeRoot + name);
        const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
          .map((value) => value.toString(16).padStart(2, "0")).join("");
        if (digest !== expected) throw new Error(`Installed WASM asset changed: ${name}`);
        hashes[name] = digest;
      }
      return hashes;
    }, runtimePackage);
    const adapterURL = new URL("runtime-test/browser.mjs", page.url()).href;
    await page.evaluate(async (spec) => {
      const pyodide = window.slicerWeb.pyodide;
      pyodide.globals.set("_runtime_spec_json", JSON.stringify(spec));
      await pyodide.runPythonAsync(`
import json
from slicerweb import jobs
_runtime_state = {"done": False, "error": None}
def _runtime_done(result, files):
    _runtime_state.update(done=True, result=result)
def _runtime_failed(error):
    _runtime_state.update(done=True, error=str(error))
_runtime_spec = json.loads(_runtime_spec_json)
jobs.run('''
import sys, sysconfig
from pathlib import Path
sys.path.insert(0, "/work")
from worker import run_runtime
result = await run_runtime(runtime_adapter_url, runtime_options)
assert Path(result["application"]["module_path"]).is_relative_to(Path(sysconfig.get_paths()["purelib"]) / "slicer_home")
''', globals=_runtime_spec["globals"],
files={name: text.encode() for name, text in _runtime_spec["files"].items()},
packages=["scipy"], onDone=_runtime_done, onFailed=_runtime_failed)
`);
    }, { globals: { runtime_adapter_url: adapterURL, runtime_options: {
      allow_software: values["allow-software"], require_f16: values["require-f16"],
    } }, files: workerFiles });
    await page.waitForFunction(() => window.slicerWeb.pyodide.runPython('_runtime_state["done"]'),
      null, { timeout: 300000 });
    const state = await page.evaluate(() => JSON.parse(
      window.slicerWeb.pyodide.runPython("json.dumps(_runtime_state)")));
    assert.equal(state.error, null, state.error);
    assert.equal(state.result.status, "passed");
    report.runtime = { ...state.result, installed_asset_hashes: installedHashes };
  }
  await writeFile(path.join(output, "results/slicerweb.json"), JSON.stringify(report, null, 2) + "\n");
  await page.screenshot({ path: path.join(output, "results/slicerweb.png") });
  console.log(JSON.stringify({ status: report.status, tests_run: report.tests_run,
    browser: report.browser, slicerweb_revision: report.slicerweb_revision,
    ...(report.runtime ? { adapter: report.runtime.adapter, shader_f16: report.runtime.shader_f16,
      cases_passed: report.runtime.cases.filter((entry) => entry.status === "passed").length } : {}),
  }));
} catch (error) {
  await writeFile(path.join(output, "results/slicerweb.json"), JSON.stringify({ status: "failed", error: String(error) }, null, 2));
  throw error;
} finally {
  await browser?.close();
  server.kill("SIGTERM");
}
