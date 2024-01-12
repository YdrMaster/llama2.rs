use std::iter::zip;

macro_rules! slice {
    ($blob:expr; $width:expr; [$line:expr]) => {
        $blob[$line * $width..][..$width]
    };
}

pub(crate) use slice;

pub(crate) fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32]) {
    let n = weight.len();
    let lines = x.len() / n;

    debug_assert_eq!(o.len(), x.len());
    debug_assert_eq!(x.len() % n, 0);

    for i in 0..lines {
        let o = &mut slice!(o; n; [i]);
        let x = &slice!(x; n; [i]);
        let ss = rmsnorm_reduce(x);
        zip(o, zip(x, weight)).for_each(|(o, (x, w))| *o = w * (ss * x));
    }
}

pub(crate) fn rmsnorm_inplace(x: &mut [f32], weight: &[f32]) {
    let n = weight.len();
    let lines = x.len() / n;

    debug_assert_eq!(x.len() % n, 0);

    for i in 0..lines {
        let x = &mut slice!(x; n; [i]);
        let ss = rmsnorm_reduce(x);
        zip(x, weight).for_each(|(x, w)| *x *= w * ss);
    }
}

#[inline]
fn rmsnorm_reduce(x: &[f32]) -> f32 {
    // (Σx^2 / n + δ)^(-1/2)
    let y = x.iter().map(|x| x * x).sum::<f32>();
    (y / (x.len() as f32) + 1e-5).powf(-0.5)
}

/// y = wx.
pub(crate) fn matmul(y: &mut [f32], x: &[f32], w: &[f32]) {
    let n = x.len();
    y.iter_mut().enumerate().for_each(|(i, y)| {
        *y = zip(&w[i * n..], x).map(|(&w, &x)| w * x).sum::<f32>();
    });
}

/// y += wx.
// pub(crate) fn sgemm(y: &mut [f32], x: &[f32], w: &[f32]) {
//     let n = x.len();
//     y.iter_mut().enumerate().for_each(|(i, y)| {
//         zip(&w[i * n..], x).for_each(|(&w, &x)| *y += w * x);
//     });
// }

pub(crate) fn softmax(x: &mut [f32]) {
    let max = *x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let sum = x
        .iter_mut()
        .map(|x| {
            *x = (*x - max).exp();
            *x
        })
        .sum::<f32>();
    x.iter_mut().for_each(|x| *x /= sum);
}

#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}
