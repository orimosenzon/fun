/*
 * Where a photo was taken, read from the file the phone wrote.
 *
 * Every upload path re-encodes photos on a canvas (draft.js `shrink`, store.js
 * `encodeImage`), and that throws the EXIF away. So the location is read here,
 * from the original, before anything else touches it, and travels from then on
 * as a field of its own: `at: [lat, lng]` on the photo, the same order as a
 * trail's path.
 *
 * The decoder is the one exif-test.html proved on real phones: the TIFF block
 * sits behind "Exif\0\0" in a JPEG's APP1, a WebP's EXIF chunk and a HEIC's
 * Exif item alike, so rather than walk three container formats it looks for
 * that marker near the start of the file.
 */
const Exif = (() => {
  const HEAD = 512 * 1024;

  function findTiff(bytes) {
    const n = bytes.length - 8;
    for (let i = 0; i < n; i++) {
      if (bytes[i] === 0x45 && bytes[i + 1] === 0x78 && bytes[i + 2] === 0x69 &&
          bytes[i + 3] === 0x66 && bytes[i + 4] === 0 && bytes[i + 5] === 0) {
        const t = i + 6;
        if ((bytes[t] === 0x49 && bytes[t + 1] === 0x49) ||
            (bytes[t] === 0x4d && bytes[t + 1] === 0x4d)) return t;
      }
    }
    return -1;
  }

  function parse(buf) {
    const bytes = new Uint8Array(buf);
    const base = findTiff(bytes);
    if (base < 0) return null;
    const dv = new DataView(buf);
    const le = bytes[base] === 0x49;
    const u16 = (o) => dv.getUint16(base + o, le);
    const u32 = (o) => dv.getUint32(base + o, le);
    const SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8 };

    function value(entry) {
      const type = u16(entry + 2), count = u32(entry + 4);
      const size = (SIZE[type] || 1) * count;
      const at = size > 4 ? u32(entry + 8) : entry + 8;
      if (type === 2) {
        let s = '';
        for (let i = 0; i < count; i++) {
          const c = bytes[base + at + i];
          if (!c) break;
          s += String.fromCharCode(c);
        }
        return s.trim();
      }
      const out = [];
      for (let i = 0; i < count; i++) {
        if (type === 3) out.push(u16(at + i * 2));
        else if (type === 4) out.push(u32(at + i * 4));
        else if (type === 5) {
          const d = u32(at + i * 8 + 4);
          out.push(d ? u32(at + i * 8) / d : 0);
        } else out.push(bytes[base + at + i]);
      }
      return count === 1 ? out[0] : out;
    }

    function ifd(off) {
      const tags = {};
      if (!off || off + 2 > bytes.length - base) return tags;
      const count = u16(off);
      for (let i = 0; i < count; i++) {
        const e = off + 2 + i * 12;
        try { tags[u16(e)] = value(e); } catch (err) { /* a tag past the end */ }
      }
      return tags;
    }

    const ifd0 = ifd(u32(4));
    const sub = ifd0[0x8769] ? ifd(ifd0[0x8769]) : {};
    const gps = ifd0[0x8825] ? ifd(ifd0[0x8825]) : null;
    return { ifd0, sub, gps };
  }

  const dms = (v) => Array.isArray(v) ? v[0] + v[1] / 60 + (v[2] || 0) / 3600 : NaN;

  /** `{ at: [lat, lng] | null, redacted, taken: 'YYYY:MM:DD hh:mm:ss' | null }`,
   *  `redacted` meaning the phone emptied the GPS block on the way here; or null
   *  for a file with no EXIF at all. Never throws: a photo without a location
   *  is still a photo. */
  async function read(blob) {
    try {
      const exif = parse(await blob.slice(0, HEAD).arrayBuffer());
      if (!exif) return null;
      let at = null, redacted = false;
      if (exif.gps) {
        let lat = dms(exif.gps[2]), lng = dms(exif.gps[4]);
        if (exif.gps[1] === 'S') lat = -lat;
        if (exif.gps[3] === 'W') lng = -lng;
        // Android's picker, for an app without ACCESS_MEDIA_LOCATION, keeps the
        // GPS block and zeroes what is in it. Zero is not a place anyone walks.
        if (isFinite(lat) && isFinite(lng) && (lat || lng)) {
          at = [+lat.toFixed(6), +lng.toFixed(6)];
        } else redacted = true;
      }
      return { at, redacted, taken: exif.sub[0x9003] || exif.ifd0[0x132] || null };
    } catch (err) {
      return null;
    }
  }

  /** Just the location, or null. */
  const locate = async (blob) => ((await read(blob)) || {}).at || null;

  return { read, locate };
})();
