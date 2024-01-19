use crate::kernel::slice;
use std::{fmt::Display, fs::File, io::Write, ops::Add, path::PathBuf};

pub trait Logger {
    fn log<T: Display>(&mut self, title: &[&str], buf: &[T], shape: &[usize]);
}

impl Logger for () {
    fn log<T: Display>(&mut self, _: &[&str], _: &[T], _: &[usize]) {}
}

pub struct FsLogger(PathBuf);

impl FsLogger {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        if path.is_dir() {
            std::fs::remove_dir_all(&path).unwrap();
        } else if path.is_file() {
            panic!("{} is a file", path.display());
        }
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Logger for FsLogger {
    fn log<T: Display>(&mut self, title: &[&str], buf: &[T], shape: &[usize]) {
        let path = self.0.join(title.join("_").add(".log"));
        write_log(&mut File::create(path).unwrap(), buf, shape).unwrap();
    }
}

fn write_log<T: Display>(to: &mut impl Write, buf: &[T], shape: &[usize]) -> std::io::Result<()> {
    match shape {
        [] => {
            writeln!(to, "<>")?;
            write_matrix(to, buf, (1, 1))
        }
        [len] => {
            writeln!(to, "<{len}>")?;
            write_matrix(to, buf, (*len, 1))
        }
        [rows, cols] => {
            writeln!(to, "<{rows}x{cols}>")?;
            write_matrix(to, buf, (*rows, *cols))
        }
        [batch @ .., rows, cols] => {
            let mut strides = vec![1usize; batch.len()];
            for i in (1..batch.len()).rev() {
                strides[i - 1] = strides[i] * batch[i];
            }
            let strides = strides.as_slice();
            for i in 0..batch[0] * strides[0] {
                let mut which = vec![0usize; strides.len()];
                let mut rem = i;
                for (j, &stride) in strides.iter().enumerate() {
                    which[j] = rem / stride;
                    rem %= stride;
                }
                writeln!(
                    to,
                    "<{rows}x{cols}>[{}]",
                    which
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(", "),
                )?;
                write_matrix(to, &slice!(buf; rows * cols; [i]), (*rows, *cols))?;
            }
            Ok(())
        }
    }
}

fn write_matrix<T: Display>(
    to: &mut impl Write,
    buf: &[T],
    shape: (usize, usize),
) -> std::io::Result<()> {
    let (rows, cols) = shape;
    for r in 0..rows {
        let row = &slice!(buf; cols; [r]);
        for it in row {
            write!(to, "{it:>9.6} ")?;
        }
        writeln!(to)?;
    }
    Ok(())
}

#[test]
fn test_log() {
    let array = [
        1., 2., 3., //
        4., 5., 6., //
        7., 8., 9., //
        10., 11., 12., //
    ];
    write_log(&mut std::io::stdout(), &array, &[2, 2, 3]).unwrap();
}
