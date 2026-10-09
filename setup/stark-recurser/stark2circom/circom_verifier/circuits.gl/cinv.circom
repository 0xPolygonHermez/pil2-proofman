pragma circom 2.1.0;
pragma custom_templates;

include "cmul.circom";

// Calculate the inverse of an element in Fp³ using (X³ - X - 1) as a irreductible polynomial
// The coefficients of the inverse are calculated beforehand and the resulting expression is explicitly written
// The inverse of a = a + bX + cX² is the following:
// ( (-a² - 2ac + bc + b² + c²) + (ab - c²)X + (-b² + ca + c²)X² ) * 1/(-a³ + ab² - b³ - 2a²c - 3abc - ac² + bc² - c³) 
template CInv() {
    signal input in[3];
    signal output out[3];

    var aa = in[0] * in[0];
    var ac = in[0] * in[2];
    var ba = in[1] * in[0];
    var bb = in[1] * in[1];
    var bc = in[1] * in[2];
    var cc = in[2] * in[2];

    var aaa = aa * in[0];
    var aac = aa * in[2];
    var abc = ba * in[2];
    var abb = ba * in[1];
    var acc = ac * in[2];
    var bbb = bb * in[1];
    var bcc = bc * in[2];
    var ccc = cc * in[2];

    var t = -aaa -aac-aac +abc+abc+abc + abb - acc - bbb + bcc - ccc;
    var tinv = 1/t;

    out[0] <--  (-aa -ac-ac +bc + bb - cc)*tinv;
    out[1] <--  (ba -cc)*tinv;
    out[2] <--  (-bb +ac + cc)*tinv;

    signal check[3] <== CMul()(in, out);
    check === [1,0,0];
}

// a / b in Fp³: the quotient is a hint, checked with a single CMul (b must be nonzero).
template CDiv() {
    signal input a[3];
    signal input b[3];
    signal output out[3];

    var aa = b[0] * b[0];
    var ac = b[0] * b[2];
    var ba = b[1] * b[0];
    var bb = b[1] * b[1];
    var bc = b[1] * b[2];
    var cc = b[2] * b[2];

    var t = -aa*b[0] -aa*b[2]-aa*b[2] +bc*b[0]+bc*b[0]+bc*b[0] + ba*b[1] - ac*b[2] - bb*b[1] + bc*b[2] - cc*b[2];
    var tinv = 1/t;
    var inv[3] = [(-aa -ac-ac +bc + bb - cc)*tinv, (ba -cc)*tinv, (-bb +ac + cc)*tinv];

    var A = (a[0] + a[1]) * (inv[0] + inv[1]);
    var B = (a[0] + a[2]) * (inv[0] + inv[2]);
    var C = (a[1] + a[2]) * (inv[1] + inv[2]);
    var D = a[0] * inv[0];
    var E = a[1] * inv[1];
    var F = a[2] * inv[2];
    out[0] <-- C + D - E - F;
    out[1] <-- A + C - E - E - D;
    out[2] <-- B - D + E;

    signal check[3] <== CMul()(out, b);
    check === a;
}
