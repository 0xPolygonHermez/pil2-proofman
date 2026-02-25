(async function () {
    async function compute(input) {
        return {
            doubled: input.value * 2,
            message: `Processed ${input.name}`
        };
    }
    // Get generic input from Rust (any JSON object)
    const input = Deno.core.ops.op_get_input();
    const result = await compute(input);
    // Return generic output to Rust (any JSON object)
    Deno.core.ops.op_store_result(result);
})()
