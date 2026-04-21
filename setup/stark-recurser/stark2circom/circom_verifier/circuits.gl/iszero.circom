pragma circom 2.1.0;

/*
    Check if a value is zero or not. Used in recursive2
*/
template IsZero() {
    signal input in;
    signal output {binary} out;

    signal inv;

    inv <-- in != 0 ? 1/in : 0;

    out <== 1 - in*inv;
    in*out === 0;
}