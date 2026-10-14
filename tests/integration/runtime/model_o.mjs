// Production kernels supplied by Python from the installed extension package.
function check(condition, message) { if (!condition) throw new Error(message); }
async function bytes(base, name) {
  const response = await fetch(new URL(name, base));
  check(response.ok, `Missing ModelO fixture ${name}`);
  return response.arrayBuffer();
}
async function read(device, buffer, size) {
  const staging = device.createBuffer({ size: buffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const result = staging.getMappedRange().slice(0, size);
    staging.unmap();
    return result;
  } finally { staging.destroy(); }
}
function bind(shader, storage) {
  const parameters = shader.metadata.parameters;
  check(Object.keys(storage).length === parameters.length, "Shader resource count mismatch");
  return parameters.map(parameter => {
    check(parameter.name in storage, `Missing shader buffer ${parameter.name}`);
    check((parameter.binding.space ?? 0) === 0, "Unexpected shader group");
    return { binding: parameter.binding.index, buffer: storage[parameter.name] };
  });
}
function tensorSum(data, dtype) {
  if (dtype === "int64") {
    const view = new DataView(data);
    let sum = 0;
    for (let offset = 0; offset < data.byteLength; offset += 8) sum += Number(view.getBigInt64(offset, true));
    return sum;
  }
  const values = dtype === "float32" ? new Float32Array(data) : new Uint8Array(data);
  return values.reduce((sum, value) => sum + value, 0);
}
export async function runModelO(device, base, shaders, kernel, ort) {
  check(shaders?.patch_inputs, "Installed Python did not supply Slang shaders");
  const manifest = await (await fetch(new URL("model_o.json", base))).json();
  const buffers = [];
  let session;
  const storage = size => {
    const buffer = device.createBuffer({ size: Math.ceil(size / 16) * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    buffers.push(buffer);
    return buffer;
  };
  device.pushErrorScope("validation");
  try {
    const resources = { config: storage((await bytes(base, manifest.cases[0].config)).byteLength),
      volume_zyx: storage(manifest.volume_shape.reduce((a, b) => a * b) * 4),
      ray_geometry: storage(manifest.rays * 28) };
    for (const output of manifest.cases[0].outputs) resources[output.name] = storage(output.bytes);
    device.queue.writeBuffer(resources.ray_geometry, 0, await bytes(base, manifest.geometry));
    const shader = shaders.patch_inputs;
    const compute = kernel(device, shader.code, shader.metadata.entryPoints[0].name, bind(shader, resources), shader.metadata.entryPoints[0].threadGroupSize[0]);
    session = await ort.InferenceSession.create(new Uint8Array(await bytes(base, "tensor_bridge.onnx")), {
      executionProviders: [{ name: "webgpu", device, enableInt64: true }],
      preferredOutputLocation: "gpu-buffer", enableGraphCapture: false,
      graphOptimizationLevel: "basic", extra: { session: { disable_cpu_ep_fallback: "1", disable_prepacking: "1" } },
    });
    const feeds = {}, fetches = {}, totals = {};
    for (const output of manifest.cases[0].outputs) {
      feeds[output.name] = ort.Tensor.fromGpuBuffer(resources[output.name], {
        dataType: output.dtype, dims: output.shape });
      totals[output.name] = storage(4);
      fetches[output.name + "_total"] = ort.Tensor.fromGpuBuffer(totals[output.name], {
        dataType: "float32", dims: [] });
    }
    const results = [];
    for (const spec of manifest.cases) {
      device.queue.writeBuffer(resources.volume_zyx, 0, await bytes(base, spec.volume));
      device.queue.writeBuffer(resources.config, 0, await bytes(base, spec.config));
      compute(manifest.rays + 48**3);
      const outputs = await session.run(feeds, fetches, {
        extra: { disable_synchronize_execution_providers: "1" },
      });
      const errors = {};
      for (const output of spec.outputs) {
        check(outputs[output.name + "_total"].gpuBuffer === totals[output.name], "ORT replaced the caller output buffer");
        const actualBytes = await read(device, resources[output.name], output.bytes);
        const total = new Float32Array(await read(device, totals[output.name], 4))[0];
        const expectedTotal = tensorSum(actualBytes, output.dtype);
        check(Number.isFinite(total) && Math.abs(total - expectedTotal) <= 0.002 + 2e-4 * Math.abs(expectedTotal),
          `ORT did not read production buffer ${output.name}: ${total} != ${expectedTotal}`);
        const expectedBytes = await bytes(base, output.expected);
        const floats = output.dtype === "float32";
        const actual = floats ? new Float32Array(actualBytes) : new Uint8Array(actualBytes);
        const expected = floats ? new Float32Array(expectedBytes) : new Uint8Array(expectedBytes);
        let maximum = 0;
        for (let index = 0; index < actual.length; index++) {
          const delta = Math.abs(actual[index] - expected[index]);
          check(Number.isFinite(actual[index]), `${spec.name}/${output.name}: nonfinite output`);
          maximum = Math.max(maximum, delta);
          check(delta <= (floats ? output.atol + output.rtol * Math.abs(expected[index]) : 0),
            `${spec.name}/${output.name}[${index}]: ${actual[index]} != ${expected[index]}`);
        }
        errors[output.name] = maximum;
      }
      results.push({ name: spec.name, status: "passed", max_absolute_error: errors });
    }
    let trained = null;
    if (manifest.model) {
      await session.release();
      session = null;
      const model = await bytes(base, manifest.model.file);
      const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", model)))
        .map(value => value.toString(16).padStart(2, "0")).join("");
      check(digest === manifest.model.sha256, "Trained model digest mismatch");
      session = await ort.InferenceSession.create(new Uint8Array(model), {
        executionProviders: [{ name: "webgpu", device, enableInt64: true,
          preferredLayout: "NCHW", storageBufferCacheMode: "simple" }],
        preferredOutputLocation: "gpu-buffer", enableGraphCapture: false,
        graphOptimizationLevel: "basic", extra: { session: { disable_cpu_ep_fallback: "1", disable_prepacking: "1" } },
      });
      const logits = storage(48**3 * 4);
      const output = ort.Tensor.fromGpuBuffer(logits, { dataType: "float32", dims: [1, 1, 48, 48, 48] });
      const cases = [];
      let thresholdFixture;
      for (const spec of manifest.cases) {
        device.queue.writeBuffer(resources.volume_zyx, 0, await bytes(base, spec.volume));
        device.queue.writeBuffer(resources.config, 0, await bytes(base, spec.config));
        compute(manifest.rays + 48**3);
        const outputs = await session.run(feeds, { mask_logits: output }, {
          extra: { disable_synchronize_execution_providers: "1" },
        });
        check(outputs.mask_logits.gpuBuffer === logits, "ORT replaced trained output buffer");
        const actual = new Float32Array(await read(device, logits, 48**3 * 4));
        const expected = new Float32Array(await bytes(base, spec.model_logits));
        let maximum = 0;
        actual.forEach((value, index) => {
          const error = Math.abs(value - expected[index]);
          maximum = Math.max(maximum, error);
          check(Number.isFinite(value) && error <= manifest.model.atol + manifest.model.rtol * Math.abs(expected[index]),
            `${spec.name}: trained logit mismatch at ${index}`);
          check((value > 0) === (expected[index] > 0), `${spec.name}: trained mask mismatch at ${index}`);
        });
        cases.push({ name: spec.name, max_absolute_error: maximum, mask_differences: 0,
          foreground_voxels: expected.reduce((total, value) => total + Number(value > 0), 0) });
        thresholdFixture = { logits: Array.from(actual), mask: Array.from(expected, value => Number(value > 0)) };
      }
      trained = { status: "passed", model_sha256: digest, cases, cpu_fallback_disabled: true,
        logit_tolerance: { atol: manifest.model.atol, rtol: manifest.model.rtol },
        threshold_fixture: thresholdFixture };
    }
    const values = new Float32Array(await bytes(base, "logits.bin"));
    const logits = storage(values.byteLength);
    const expected = new Uint8Array(await bytes(base, "mask.bin"));
    device.queue.writeBuffer(logits, 0, values);
    const actual = new Float32Array(await read(device, logits, values.byteLength));
    actual.forEach((value, index) => check(Number.isFinite(value)
      && Number(value > 0) === expected[index], `Logit threshold mismatch at ${index}`));
    return { status: "passed", cases: results, caller_buffers_reused: true,
      ort_reads_all_seven_inputs: true,
      trained_model: trained,
      threshold_fixture: { logits: Array.from(actual), mask: Array.from(expected) } };

  } finally {
    await device.queue.onSubmittedWorkDone();
    if (session) await session.release();
    for (const buffer of buffers) buffer.destroy();
    const error = await device.popErrorScope();
    if (error) throw new Error(error.message);
  }
}
