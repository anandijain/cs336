use super::{Op, Value};
use std::{
    collections::HashSet,
    fmt::Write as _,
    fs, io,
    path::{Path, PathBuf},
    process::Command,
    rc::Rc,
};


impl Value {
    fn dot_id(&self) -> usize {
        Rc::as_ptr(&self.node) as usize
    }

    fn to_dot(&self) -> String {
        let mut dot = String::from(
            "digraph computation {\n\
             \trankdir=LR;\n\
             \tgraph [bgcolor=\"white\"];\n\
             \tnode [fontname=\"Helvetica\"];\n",
        );
        let mut visited = HashSet::new();
        self.write_dot(&mut dot, &mut visited);
        dot.push_str("}\n");
        dot
    }

    fn write_dot(&self, dot: &mut String, visited: &mut HashSet<usize>) {
        let id = self.dot_id();
        if !visited.insert(id) {
            return;
        }

        // Release the RefCell borrow before recursing into predecessor nodes.
        let (data, grad, op, prev) = {
            let node = self.node.borrow();
            (node.data, node.grad, node.op, node.prev.clone())
        };

        writeln!(
            dot,
            "\tvalue_{id:x} [label=\"{{ data {data:.4} | grad {grad:.4} }}\", shape=record];"
        )
        .expect("writing to a String cannot fail");

        if !matches!(op, Op::Leaf) {
            writeln!(
                dot,
                "\top_{id:x} [label=\"{}\", shape=circle, width=0.35, fixedsize=true];",
                op.symbol()
            )
            .expect("writing to a String cannot fail");
            writeln!(dot, "\top_{id:x} -> value_{id:x};").expect("writing to a String cannot fail");

            for predecessor in &prev {
                predecessor.write_dot(dot, visited);
                writeln!(dot, "\tvalue_{:x} -> op_{id:x};", predecessor.dot_id())
                    .expect("writing to a String cannot fail");
            }
        }
    }
}

pub(super) fn render_graph(root: &Value, name: &str) -> io::Result<PathBuf> {
    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("graphs");
    fs::create_dir_all(&output_dir)?;

    let dot_path = output_dir.join(format!("{name}.dot"));
    let svg_path = output_dir.join(format!("{name}.svg"));
    fs::write(&dot_path, root.to_dot())?;

    let output = Command::new("dot")
        .arg("-Tsvg")
        .arg(&dot_path)
        .arg("-o")
        .arg(&svg_path)
        .output()?;

    if !output.status.success() {
        return Err(io::Error::other(format!(
            "Graphviz failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    Ok(svg_path)
}
