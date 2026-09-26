#!/usr/bin/env python3
"""Generates crates/opusorus/src/celt/static_modes_fixed.rs from
vendor/libopus/celt/static_modes_fixed.h (fixed-point static CELT modes).

The header is preprocessed twice (with and without ENABLE_QEXT); tables that differ become
`#[cfg(feature = "qext")]` / `#[cfg(not(feature = "qext"))]` pairs. Integer tables that are
identical to the float header's (logN, pulse caches, FFT bit-reversal tables) are not emitted:
`static_modes.rs` shares them between both builds (checked here, value by value).

Usage: python3 scripts/gen_static_modes_fixed.py   (from the repository root)
"""

import re
import sys

ROOT = '.'
FIXED_H = ROOT + '/vendor/libopus/celt/static_modes_fixed.h'
FLOAT_H = ROOT + '/vendor/libopus/celt/static_modes_float.h'
OUT = ROOT + '/crates/opusorus/src/celt/static_modes_fixed.rs'


def preprocess(text, defined):
    """Minimal preprocessor: #ifdef/#ifndef/#else/#endif/#define/#include only."""
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
    defined = set(defined)
    out = []
    stack = []  # (active_before, this_branch_active)
    active = True
    for line in text.split('\n'):
        s = line.strip()
        m = re.match(r'#\s*(ifdef|ifndef)\s+(\w+)', s)
        if m:
            cond = (m.group(2) in defined) == (m.group(1) == 'ifdef')
            stack.append((active, cond))
            active = active and cond
            continue
        if re.match(r'#\s*else', s):
            parent, cond = stack[-1]
            stack[-1] = (parent, not cond)
            active = parent and not cond
            continue
        if re.match(r'#\s*endif', s):
            parent, _ = stack.pop()
            active = parent
            continue
        if not active:
            continue
        m = re.match(r'#\s*define\s+(\w+)', s)
        if m:
            defined.add(m.group(1))
            continue
        if s.startswith('#'):
            if re.match(r'#\s*include', s):
                continue
            raise SystemExit('unhandled directive: ' + s)
        out.append(line)
    assert not stack
    return '\n'.join(out)


def parse_init(tokens, i):
    """Parses an initializer starting at tokens[i]; returns (value, next_index)."""
    if tokens[i] == '{':
        i += 1
        items = []
        while tokens[i] != '}':
            v, i = parse_init(tokens, i)
            items.append(v)
            if tokens[i] == ',':
                i += 1
        return items, i + 1
    if tokens[i] == '&':
        return ('ref', tokens[i + 1]), i + 2
    if tokens[i] == '-':
        return -int(tokens[i + 1]), i + 2
    t = tokens[i]
    if re.fullmatch(r'\d+', t):
        return int(t), i + 1
    return ('id', t), i + 1


def declarations(text):
    decls = {}
    order = []
    for m in re.finditer(r'static\s+const\s+([\w ]+?)\s*(\*\s*const\s+)?(\w+)\s*(\[(\w*)\])?\s*=\s*', text):
        ctype = m.group(1).strip()
        name = m.group(3)
        start = m.end()
        # tokenise until the matching '};'
        depth = 0
        j = start
        while True:
            c = text[j]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = text[start:j + 1]
        tokens = re.findall(r'\d+|\w+|[{}&,\-]', body)
        val, _ = parse_init(tokens, 0)
        decls[name] = (ctype, val)
        order.append(name)
    return decls, order


RUST_NAME = {}


def rname(c):
    return c.upper()


def fmt_list(vals, per_line=None, indent='    '):
    items = [str(v) for v in vals]
    lines = []
    cur = indent
    for it in items:
        piece = it + ','
        if len(cur) + len(piece) + 1 > 100 and cur.strip():
            lines.append(cur.rstrip())
            cur = indent
        cur += piece + ' '
    if cur.strip():
        lines.append(cur.rstrip())
    return '\n'.join(lines)


def emit(name, ctype, val, cfg):
    out = []
    attr = ''
    if cfg:
        attr = '#[cfg(%s)]\n' % cfg
    rn = rname(name)
    if ctype in ('celt_coef', 'opus_int16', 'unsigned char'):
        rty = {'celt_coef': 'CeltCoef', 'opus_int16': 'i16', 'unsigned char': 'u8'}[ctype]
        out.append('%s#[rustfmt::skip]\npub static %s: [%s; %d] = [\n%s\n];\n' % (
            attr, rn, rty, len(val), fmt_list(val)))
    elif ctype == 'kiss_twiddle_cpx':
        body = '\n'.join('    KissTwiddleCpx { r: %d, i: %d },' % (r, i) for r, i in val)
        out.append('%s#[rustfmt::skip]\npub static %s: [KissTwiddleCpx; %d] = [\n%s\n];\n' % (
            attr, rn, len(val), body))
    elif ctype == 'kiss_fft_state':
        nfft, scale, scale_shift, shift, factors, bitrev, twiddles, _arch = val
        out.append('''%spub static %s: KissFftState = KissFftState {
    nfft: %d,
    scale: %d,
    scale_shift: %d,
    shift: %d,
    factors: [%s],
    bitrev: Cow::Borrowed(&%s),
    twiddles: Cow::Borrowed(&%s),
};
''' % (attr, rn, nfft, scale, scale_shift, shift, ', '.join(str(f) for f in factors),
            rname(bitrev[1]), rname(twiddles[1])))
    elif ctype == 'CELTMode':
        (fs, overlap, nbe, effe, preemph, ebands, maxlm, nbshort, shortsize, nballoc, allocv,
         logn, window, mdct, cache) = val[:15]
        qext_cache = val[15] if len(val) > 15 else None
        n, maxshift, states, trig = mdct
        s = '''%spub static %s: CeltMode = CeltMode {
    fs: %d,
    overlap: %d,
    nb_ebands: %d,
    eff_ebands: %d,
    preemph: [%s],
    e_bands: Cow::Borrowed(&%s),
    max_lm: %d,
    nb_short_mdcts: %d,
    short_mdct_size: %d,
    nb_alloc_vectors: %d,
    alloc_vectors: Cow::Borrowed(&%s),
    log_n: Cow::Borrowed(&%s),
    window: Cow::Borrowed(&%s),
    mdct: MdctLookup {
        n: %d,
        maxshift: %d,
        kfft: [
%s
        ],
        trig: Cow::Borrowed(&%s),
    },
    cache: PulseCache {
        size: %d,
        index: Cow::Borrowed(&%s),
        bits: Cow::Borrowed(&%s),
        caps: Cow::Borrowed(&%s),
    },
''' % (attr, rn, fs, overlap, nbe, effe, ', '.join(str(p) for p in preemph), rname(ebands[1]),
            maxlm, nbshort, shortsize, nballoc, rname(allocv[1]), rname(logn[1]),
            rname(window[1]), n, maxshift,
            '\n'.join('            Cow::Borrowed(&%s),' % rname(st[1]) for st in states),
            rname(trig[1]), cache[0], rname(cache[1][1]), rname(cache[2][1]),
            rname(cache[3][1]))
        # qext_cache is present in every QEXT build (it is a CeltMode field).
        if qext_cache is not None:
            s += '''    #[cfg(feature = "qext")]
    qext_cache: PulseCache {
        size: %d,
        index: Cow::Borrowed(&%s),
        bits: Cow::Borrowed(&%s),
        caps: Cow::Borrowed(&%s),
    },
''' % (qext_cache[0], rname(qext_cache[1][1]), rname(qext_cache[2][1]),
                rname(qext_cache[3][1]))
        s += '};\n'
        out.append(s)
    elif ctype == 'CELTMode * const' or ctype == 'CELTMode':
        raise SystemExit('unexpected')
    else:
        raise SystemExit('unhandled type %s for %s' % (ctype, name))
    return ''.join(out)


def main():
    fixed_src = open(FIXED_H).read()
    float_src = open(FLOAT_H).read()
    fx_q, order_q = declarations(preprocess(fixed_src, {'FIXED_POINT', 'ENABLE_QEXT'}))
    fx_n, order_n = declarations(preprocess(fixed_src, {'FIXED_POINT'}))
    fl_q, _ = declarations(preprocess(float_src, {'ENABLE_QEXT'}))
    fl_n, _ = declarations(preprocess(float_src, set()))

    names = []
    for n in order_q + order_n:
        if n not in names and n != 'static_mode_list':
            names.append(n)

    shared = []
    parts = []
    for name in names:
        in_q = name in fx_q
        in_n = name in fx_n
        ctype = (fx_q.get(name) or fx_n.get(name))[0]
        # Integer tables identical to the float header's are shared via static_modes.rs.
        if ctype in ('opus_int16', 'unsigned char'):
            for fx, fl in ((fx_q, fl_q), (fx_n, fl_n)):
                if name in fx:
                    assert name in fl and fl[name][1] == fx[name][1], name
            shared.append(name)
            continue
        same = in_q and in_n and fx_q[name] == fx_n[name]
        # A mode struct only differs by its (cfg'd) `qext_cache` field.
        if ctype == 'CELTMode' and in_q and in_n:
            same = fx_q[name][1][:15] == fx_n[name][1][:15]
        if same:
            parts.append(emit(name, ctype, fx_q[name][1], None))
        else:
            if in_q:
                parts.append(emit(name, ctype, fx_q[name][1], 'feature = "qext"'))
            if in_n:
                parts.append(emit(name, ctype, fx_n[name][1], 'not(feature = "qext")'))

    shared_doc = ''
    cur = '//!'
    for word in ('The integer tables that are identical in the float header (' +
                 ', '.join('`%s`' % n for n in shared) +
                 ') are shared from [`super::static_modes`] (checked by the generator).').split(' '):
        if len(cur) + 1 + len(word) > 100:
            shared_doc += cur + '\n'
            cur = '//!'
        cur += ' ' + word
    shared_doc += cur + '\n'
    header = '''//! Port of `celt/static_modes_fixed.h`: the statically-defined CELT modes of the fixed-point
//! build (feature `fixed-point`; the Q31 `celt_coef` variants with `qext`).
//!
//! Generated by `scripts/gen_static_modes_fixed.py` from the C header; values are copied verbatim.
%s
#![allow(
    missing_docs,
    reason = "generated tables mirror the C header one-to-one"
)]

use alloc::borrow::Cow;

use super::arch::CeltCoef;
#[cfg(feature = "qext")]
use super::static_modes::{
    FFT_BITREV960, QEXT_CACHE_BITS50, QEXT_CACHE_CAPS50, QEXT_CACHE_INDEX50,
};
use super::static_modes::{
    BAND_ALLOCATION, CACHE_BITS50, CACHE_CAPS50, CACHE_INDEX50, CeltMode, EBAND5MS, FFT_BITREV60,
    FFT_BITREV120, FFT_BITREV240, FFT_BITREV480, KissFftState, KissTwiddleCpx, LOGN400,
    MdctLookup, PulseCache,
};

''' % shared_doc
    footer = '''/// `static_mode_list`: all statically-defined modes.
#[cfg(not(feature = "qext"))]
pub static STATIC_MODE_LIST: [&CeltMode; 1] = [&MODE48000_960_120];
/// `static_mode_list`: all statically-defined modes.
#[cfg(feature = "qext")]
pub static STATIC_MODE_LIST: [&CeltMode; 2] = [&MODE48000_960_120, &MODE96000_1920_240];
'''
    open(OUT, 'w').write(header + '\n'.join(parts) + '\n' + footer)
    print('wrote', OUT, file=sys.stderr)


if __name__ == '__main__':
    main()
