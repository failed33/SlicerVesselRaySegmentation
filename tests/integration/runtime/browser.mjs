// GPU adapter called from Python inside the SlicerWeb job worker.
import * as ort from "./ort.webgpu.min.mjs";

function check(condition, message) {
  if (!condition) throw new Error(message);
}

async function fetchBytes(name) {
  const response = await fetch(new URL(name, import.meta.url));
  check(response.ok, `Cannot fetch ${name}: ${response.status}`);
  return response.arrayBuffer();
}

async function sha256(bytes) {
  return Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((value) => value.toString(16).padStart(2, "0")).join("");
}

function kernel(device, code, entryPoint, buffers) {
  const pipeline = device.createComputePipeline({ layout: "auto", compute: {
    module: device.createShaderModule({ code }), entryPoint,
  } });
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  return (elements) => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(elements / 64));
    pass.end();
    device.queue.submit([encoder.finish()]);
  };
}

async function runCase(device, spec, counters) {
  const buffers = [];
  let session;
  const storage = (bytes) => {
    const buffer = device.createBuffer({ size: Math.ceil(bytes / 16) * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    buffers.push(buffer);
    return buffer;
  };
  device.pushErrorScope("validation");
  try {
    const model = await fetchBytes(spec.model);
    const hash = await sha256(model);
    check(hash === spec.sha256, "Browser/native model fixture hashes differ");
    session = await ort.InferenceSession.create(new Uint8Array(model), {
      executionProviders: [{ name: "webgpu", device, preferredLayout: "NCHW" }],
      preferredOutputLocation: "gpu-buffer", enableGraphCapture: false,
      graphOptimizationLevel: "basic", extra: { session: { disable_cpu_ep_fallback: "1" } },
    });
    const itemsize = spec.dtype === "float16" ? 2 : 4;
    const x = storage(spec.input_elements * itemsize);
    const y = storage(spec.output_elements * itemsize);
    const audit = storage(spec.output_elements * spec.iterations * 4);
    const input = ort.Tensor.fromGpuBuffer(x, { dataType: spec.dtype, dims: spec.input_shape });
    const output = ort.Tensor.fromGpuBuffer(y, { dataType: spec.dtype, dims: spec.output_shape });
    const produce = kernel(device, spec.producer, "produce", [x]);
    const consume = kernel(device, spec.consumer, "consume", [y, x, audit]);
    counters.inputs = new Set([x]);
    const before = { waits: counters.waits, writes: counters.inputWrites };
    for (let step = 0; step < spec.iterations; step++) {
      produce(spec.input_elements);
      const outputs = await session.run({ x: input }, { y: output }, {
        extra: { disable_synchronize_execution_providers: "1" },
      });
      check(outputs.y.location === "gpu-buffer", "ORT output left the GPU");
      check(outputs.y.gpuBuffer === y, "ORT replaced the caller-owned output buffer");
      consume(spec.output_elements);
    }
    const waits = counters.waits - before.waits;
    const writes = counters.inputWrites - before.writes;
    check(waits === 0, `${waits} intermediate GPU completion waits`);
    check(writes === 0, "ORT uploaded shader-produced input from the CPU");
    const readback = device.createBuffer({ size: audit.size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    buffers.push(readback);
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(audit, 0, readback, 0, audit.size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = Array.from(new Float32Array(readback.getMappedRange()).slice(0, spec.expected.length));
    readback.unmap();
    actual.forEach((value, index) => check(Number.isFinite(value)
      && Math.abs(value - spec.expected[index]) <= spec.atol + spec.rtol * Math.abs(spec.expected[index]),
    `${spec.name} result ${index}: ${value}, expected ${spec.expected[index]}`));
    check(actual.slice(0, spec.output_elements).some((value, index) =>
      value !== actual[actual.length - spec.output_elements + index]), "Output did not depend on changed input");
    return { name: spec.name, status: "passed", model_sha256: hash, dtype: spec.dtype,
      input_shape: spec.input_shape, output_shape: spec.output_shape, iterations: spec.iterations,
      cpu_fallback_disabled: true, caller_output_buffer_retained: true,
      pipeline_host_waits: waits, input_upload_bytes: writes, final_readbacks: 1, values: actual };
  } finally {
    counters.inputs = new Set();
    await device.queue.onSubmittedWorkDone();
    if (session) await session.release();
    for (const buffer of buffers) buffer.destroy();
    const error = await device.popErrorScope();
    if (error) throw new Error(`WebGPU validation: ${error.message}`);
  }
}

export async function run(optionsJson = "{}") {
  const options = JSON.parse(optionsJson);
  check(navigator.gpu, "WebGPU unavailable inside SlicerWeb's Python worker");
  const manifest = JSON.parse(new TextDecoder().decode(await fetchBytes("manifest.json")));
  const required = ["ort.webgpu.min.mjs", "ort-wasm-simd-threaded.asyncify.mjs",
    "ort-wasm-simd-threaded.asyncify.wasm"];
  for (const name of required) {
    check(typeof manifest.files[name] === "string", `Runtime manifest omits ${name}`);
    check(await sha256(await fetchBytes(name)) === manifest.files[name], `Runtime file hash differs: ${name}`);
  }
  const adapter = await navigator.gpu.requestAdapter();
  check(adapter, "No WebGPU adapter available");
  const info = adapter.info;
  const fallback = info.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null;
  const software = fallback === true || /swiftshader|llvmpipe|lavapipe|software/i.test(
    [info.vendor, info.device, info.description].join(" "));
  check(options.allow_software || !software, "Software adapter requires --allow-software");
  const f16 = adapter.features.has("shader-f16");
  check(f16 || !options.require_f16, "Adapter lacks shader-f16");
  const device = await adapter.requestDevice({ requiredFeatures: f16 ? ["shader-f16"] : [] });
  const counters = { waits: 0, inputWrites: 0, inputs: new Set() };
  const wait = device.queue.onSubmittedWorkDone.bind(device.queue);
  device.queue.onSubmittedWorkDone = () => { counters.waits++; return wait(); };
  const write = device.queue.writeBuffer.bind(device.queue);
  device.queue.writeBuffer = (buffer, ...args) => {
    if (counters.inputs.has(buffer)) counters.inputWrites++;
    return write(buffer, ...args);
  };
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = new URL("./", import.meta.url).href;
  ort.env.logLevel = "warning";
  const cases = JSON.parse(new TextDecoder().decode(await fetchBytes("cases.json")));
  check(cases.length === 4, "Expected four shared ConvTranspose3D fixtures");
  try {
    const results = [];
    for (let lifecycle = 0; lifecycle < 2; lifecycle++) {
      for (const spec of cases) {
        const result = spec.dtype === "float16" && !f16
          ? { name: spec.name, status: "skipped", reason: "shader-f16 unavailable" }
          : await runCase(device, spec, counters);
        results.push({ lifecycle, ...result });
      }
    }
    return JSON.stringify({ status: "passed", ort_version: ort.env.versions.web,
      shader_f16: f16, physical_gpu_acceptance: !software, runtime_manifest: manifest,
      adapter: { vendor: info.vendor, architecture: info.architecture, device: info.device,
        description: info.description, is_fallback: fallback, software }, cases: results });
  } finally {
    await device.queue.onSubmittedWorkDone();
    device.destroy();
  }
}
