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
  check(shaders?.patch_inputs && shaders?.sigmoid, "Installed Python did not supply Slang shaders");
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
      graphOptimizationLevel: "basic", extra: { session: { disable_cpu_ep_fallback: "1" } },
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
        const mask = output.compare_mask ? new Uint8Array(await bytes(base, output.compare_mask)) : null;
        let maximum = 0;
        for (let index = 0; index < actual.length; index++) {
          const delta = Math.abs(actual[index] - expected[index]);
          check(Number.isFinite(actual[index]), `${spec.name}/${output.name}: nonfinite output`);
          if (mask && !mask[index]) {
            check(actual[index] >= 0 && actual[index] <= 1, "Flat-profile eta out of bounds");
          } else {
            maximum = Math.max(maximum, delta);
            check(delta <= (floats ? output.atol + output.rtol * Math.abs(expected[index]) : 0),
              `${spec.name}/${output.name}[${index}]: ${actual[index]} != ${expected[index]}`);
          }
        }
        errors[output.name] = maximum;
      }
      results.push({ name: spec.name, status: "passed", max_absolute_error: errors,
        known_flat_profile_eta_limit: "CPU parity excludes eta on flat profiles; values remain finite in [0,1]." });
    }
    const inputs = { logits: storage(48**3 * 4), probability: storage(48**3 * 4), invalid: storage(4) };
    const sigmoid = shaders.sigmoid;
    const consume = kernel(device, sigmoid.code, sigmoid.metadata.entryPoints[0].name, bind(sigmoid, inputs), sigmoid.metadata.entryPoints[0].threadGroupSize[0]);
    const logits = new Float32Array(await bytes(base, "logits.bin"));
    const expected = new Float32Array(await bytes(base, "probabilities.bin"));
    for (const value of [null, NaN, Infinity, -Infinity, null]) {
      const data = logits.slice();
      if (value !== null) data[data.length - 1] = value;
      device.queue.writeBuffer(inputs.logits, 0, data);
      device.queue.writeBuffer(inputs.invalid, 0, new Uint32Array([0]));
      consume(48**3);
      const actual = new Float32Array(await read(device, inputs.probability, 48**3 * 4));
      const invalid = new Uint32Array(await read(device, inputs.invalid, 4))[0];
      check(invalid === Number(value !== null), "Sigmoid nonfinite/reset flag mismatch");
      if (value === null) actual.forEach((v, i) => check(Number.isFinite(v)
        && Math.abs(v - expected[i]) <= 1e-6 + 1e-6 * expected[i], `Sigmoid mismatch at ${i}`));
    }
    return { status: "passed", cases: results, sigmoid_nonfinite_and_reset: "passed", caller_buffers_reused: true,
      ort_reads_all_seven_inputs: true };
  } finally {
    await device.queue.onSubmittedWorkDone();
    if (session) await session.release();
    for (const buffer of buffers) buffer.destroy();
    const error = await device.popErrorScope();
    if (error) throw new Error(error.message);
  }
}
