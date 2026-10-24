// פיזור מכדור מוליך מושלם (פתרון מי, Mie): σ מונוסטטי מדויק לכל תדר
// σ/πa² = |Σ (-1)^n (2n+1)(a_n - b_n)|² / (ka)²
// a_n = j_n(x)/h_n(x),  b_n = [x j_n]'/[x h_n]',  h_n = j_n + i y_n

function besselJY(x, N) {
  // j בירידה (מילר), y בעלייה
  const M = N + 20 + Math.ceil(x);
  const j = new Float64Array(M + 2);
  j[M + 1] = 0; j[M] = 1e-30;
  for (let n = M; n >= 1; n--) j[n - 1] = (2 * n + 1) / x * j[n] - j[n + 1];
  const j0 = Math.sin(x) / x, s = j0 / j[0];
  for (let n = 0; n <= M; n++) j[n] *= s;
  const y = new Float64Array(N + 2);
  y[0] = -Math.cos(x) / x; y[1] = -Math.cos(x) / (x * x) - Math.sin(x) / x;
  for (let n = 1; n <= N; n++) y[n + 1] = (2 * n + 1) / x * y[n] - y[n - 1];
  return { j, y };
}

export function sphereSigmaNorm(ka) {
  const x = ka, N = Math.ceil(x + 4 * Math.cbrt(x) + 4);
  const { j, y } = besselJY(x, N);
  let sr = 0, si = 0;
  for (let n = 1; n <= N; n++) {
    const hr = j[n], hi = y[n];
    // a_n = j/(j + i y)
    let d = hr * hr + hi * hi;
    const ar = j[n] * hr / d, ai = -j[n] * hi / d;
    // [x z_n]' = x z_{n-1} - n z_n
    const dj = x * j[n - 1] - n * j[n], dyr = x * y[n - 1] - n * y[n];
    d = dj * dj + dyr * dyr;
    const br = dj * dj / d, bi = -dj * dyr / d;
    const f = (n % 2 ? -1 : 1) * (2 * n + 1);
    sr += f * (ar - br); si += f * (ai - bi);
  }
  return (sr * sr + si * si) / (x * x);
}
