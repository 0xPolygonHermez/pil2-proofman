#[derive(Debug)]
pub struct Transcript {
    name: Option<String>,
    state: [String; 4],
    pending: Vec<String>,
    out: Vec<String>,
    h_cnt: usize,
    hi_cnt: usize,
    n2b_cnt: usize,
    last_code_printed: usize,
    code: Vec<String>,
}

impl Transcript {
    pub fn new(name: Option<String>) -> Self {
        Transcript {
            name,
            state: ["0".to_string(), "0".to_string(), "0".to_string(), "0".to_string()],
            pending: Vec::new(),
            out: Vec::new(),
            h_cnt: 0,
            hi_cnt: 0,
            n2b_cnt: 0,
            last_code_printed: 0,
            code: Vec::new(),
        }
    }

    pub fn get_field(&mut self, v: &str) {
        let field_code = format!("{} <== [{}, {}, {}];", v, self.get_fields1(), self.get_fields1(), self.get_fields1());
        self.code.push(field_code);
    }

    pub fn get_state(&mut self, v: &str) {
        let state_code = format!(
            "{} <== [{}, {}, {}, {}];",
            v,
            self.get_fields1(),
            self.get_fields1(),
            self.get_fields1(),
            self.get_fields1()
        );
        self.code.push(state_code);
    }

    pub fn update_state(&mut self) {
        let mut signal_name = "transcriptHash".to_string();
        if let Some(name) = &self.name {
            signal_name.push_str(&format!("_{}", name));
        }

        if self.h_cnt > 0 {
            let first_unused = self.hi_cnt.max(4);
            if first_unused < 12 {
                self.code.push(format!(
                    "for(var i = {}; i < 12; i++) {{\n    _ <== {}_{}[i]; // Unused transcript values\n}}",
                    first_unused,
                    signal_name,
                    self.h_cnt - 1
                ));
            }
        }

        self.code.push(format!(
            "signal {}_{}[12] <== Poseidon(12)([{}], [{}]);",
            signal_name,
            self.h_cnt,
            self.pending.join(", "),
            self.state.join(", ")
        ));
        self.h_cnt += 1;

        for i in 0..12 {
            self.out.push(format!("{}_{}[{}]", signal_name, self.h_cnt - 1, i));
        }

        for i in 0..4 {
            self.state[i] = format!("{}_{}[{}]", signal_name, self.h_cnt - 1, i);
        }

        self.pending.clear();
        self.hi_cnt = 0;
    }

    fn get_fields1(&mut self) -> String {
        if self.out.is_empty() {
            while self.pending.len() < 8 {
                self.pending.push("0".to_string());
            }
            self.update_state();
        }
        let res = self.out.remove(0);
        self.hi_cnt += 1;
        res
    }

    pub fn put(&mut self, a: &str, l: usize) {
        for i in 0..l {
            self.add1(&format!("{}[{}]", a, i));
        }
    }

    fn add1(&mut self, a: &str) {
        self.out.clear();
        self.pending.push(a.to_string());
        if self.pending.len() == 8 {
            self.update_state();
        }
    }

    pub fn get_permutations(&mut self, v: &str, n: usize, n_bits: usize, stark_struct_steps_nbits: usize) {
        let mut signal_name = "transcriptHash".to_string();
        if let Some(name) = &self.name {
            signal_name.push_str(&format!("_{}", name));
        }

        let total_bits = n * n_bits;
        let mut n2b = Vec::new();

        for _ in 0..n {
            let f = self.get_fields1();
            let n2b_var = format!("transcriptN2b_{}", self.n2b_cnt);
            self.n2b_cnt += 1;

            self.code.push(format!("signal {{binary}} {}[64] <== Num2Bits_strict()({});", n2b_var, f));
            n2b.push(n2b_var);
        }

        if self.hi_cnt < 12 {
            self.code.push(format!(
                "for(var i = {}; i < 12; i++) {{\n        _ <== {}_{}[i]; // Unused transcript values\n}}",
                self.hi_cnt,
                signal_name,
                self.h_cnt - 1
            ));
        }

        self.code.push("// From each transcript hash converted to bits, we assign those bits to queriesFRI[q] to define the query positions".to_string());
        self.code.push("var q = 0; // Query number".to_string());
        self.code.push("var b = 0; // Bit number".to_string());

        for (i, n2b_var) in n2b.iter().enumerate() {
            let n_bits = if i + 1 == n { total_bits - 63 * i } else { 63 };

            self.code.push(format!(
                "for(var j = 0; j < {}; j++) {{\n    {}[q][b] <== {}[j];\n    b++;\n    if(b == {}) {{\n        b = 0;\n        q++;\n    }}\n}}",
                n_bits, v, n2b_var, stark_struct_steps_nbits
            ));

            if n_bits == 63 {
                self.code.push(format!("_ <== {}[63]; // Unused last bit\n", n2b_var));
            } else {
                self.code.push(format!(
                    "for(var j = {}; j < 64; j++) {{\n    _ <== {}[j]; // Unused bits\n}}",
                    n_bits, n2b_var
                ));
            }
        }
    }

    pub fn get_code(&mut self) -> String {
        let new_code = self
            .code
            .iter()
            .skip(self.last_code_printed)
            .map(|line| format!("    {}", line))
            .collect::<Vec<_>>()
            .join("\n");
        self.last_code_printed = self.code.len();
        new_code
    }
}
