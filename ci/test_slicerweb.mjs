// Install the built wheel through SlicerWeb, then run the same MRML/widget tests as desktop.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [source, output] = process.argv.slice(2).map((value) => path.resolve(value));
const { chromium } = await import(pathToFileURL(path.join(source, "web/node_modules/playwright-core/index.mjs")));
const testCode = await readFile(new URL("test_module.py", import.meta.url), "utf8");
const info = JSON.parse(await readFile(path.join(source, "../wheels/build-info.json"), "utf8"));
const index = JSON.parse(await readFile(path.join(output, "wheels/index.json"), "utf8"));
assert.equal(index.extensions.length, 1);
assert.deepEqual(index.extensions[0].modules, ["VesselRaySegmentationDemo"]);
const server = spawn("npm", ["run", "preview", "--", "--host", "127.0.0.1", "--port", "4173", "--strictPort"],
  { cwd: path.join(source, "web"), stdio: "inherit" });
let browser;
await mkdir(path.join(output, "results"), { recursive: true });
try {
  const url = "http://127.0.0.1:4173/?sample=";
  for (let attempt = 0; ; attempt++) {
    try { if ((await fetch(url)).ok) break; } catch {}
    if (attempt >= 60) throw new Error("SlicerWeb preview server did not start");
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  browser = await chromium.launch({ channel: "chrome", headless: true,
    args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.setDefaultTimeout(300000);
  page.on("pageerror", (error) => console.error(String(error)));
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) console.log(`[${message.type()}] ${message.text()}`);
  });
  await page.goto(url);
  await page.waitForFunction(() => window.slicerWeb?.store?.status === "ready");
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
  await writeFile(path.join(output, "results/slicerweb.json"), JSON.stringify(report, null, 2) + "\n");
  await page.screenshot({ path: path.join(output, "results/slicerweb.png") });
  console.log(JSON.stringify(report));
} catch (error) {
  await writeFile(path.join(output, "results/slicerweb.json"), JSON.stringify({ status: "failed", error: String(error) }, null, 2));
  throw error;
} finally {
  await browser?.close();
  server.kill("SIGTERM");
}
