mod dot;

use std::{
    cell::RefCell,
    collections::HashSet,
    io,
    ops::{Add, Div, Mul, Neg, Sub, },
    rc::Rc,
};

fn f(x: f64) -> f64 {
    (3.0 * x).powi(2) + 4. * x + 5.
}

#[derive(Debug, Clone)]
struct Value {
    node: Rc<RefCell<Node>>,
}

#[derive(Debug, Clone)]
struct Node {
    data: f64,
    grad: f64,
    prev: Vec<Value>,
    op: Op,
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Leaf,
    Add,
    Mul,
    Sub,
    Div,
    Neg,
    Exp,
    Tanh,
}

impl Value {
    fn new(data: f64) -> Self {
        Self::from_prev(data, vec![], Op::Leaf)
    }

    fn from_prev(data: f64, prev: Vec<Value>, op: Op) -> Self {
        Value {
            node: Rc::new(RefCell::new(Node {
                data,
                grad: 0.,
                prev,
                op,
            })),
        }
    }

    fn data(&self) -> f64 {
        self.node.borrow().data
    }

    fn grad(&self) -> f64 {
        self.node.borrow().grad
    }

    fn op(&self) -> Op {
        self.node.borrow().op
    }

    fn chds(&self) -> Vec<Value> {
        self.node.borrow().prev.clone()
    }

    fn tanh(&self) -> Value {
        let x = Value::data(self);
        let t = x.tanh();
        Value::from_prev(t, vec![self.clone()], Op::Tanh)
    }
    /// given that self's grad is set, we want to propagate gradient info to the previous child nodes
    fn local_backwards(&self) {
        let g = self.grad();
        match Value::op(self) {
            Op::Add => {
                for chd in self.node.borrow().prev.clone() {
                    chd.node.borrow_mut().grad += 1.0 * g
                }
            }
            Op::Mul => {
                let [lhs, rhs]: [Value; 2] = self.node.borrow().prev.clone().try_into().unwrap();

                let rx = rhs.data();
                let lx = lhs.data();

                lhs.node.borrow_mut().grad += g * rx;
                rhs.node.borrow_mut().grad += g * lx;
            }
            Op::Leaf => {}
            _ => todo!(),
        }
    }

    fn backwards(&self) {
        self.node.borrow_mut().grad = 1.0;
        let vs = build_topo(&self);
        for v in vs {
            v.local_backwards();
        }
    }
}

fn build_topo(v: &Value) -> Vec<Value> {
    let mut visited = HashSet::new();
    let mut topo = vec![];
    visit(v, &mut visited, &mut topo);
    topo.reverse();
    topo
}

fn visit(v: &Value, visited: &mut HashSet<usize>, topo: &mut Vec<Value>) {
    let vptr = (v.node.as_ptr() as usize);
    if !visited.contains(&vptr) {
        visited.insert(vptr);
        for chd in v.chds() {
            visit(&chd, visited, topo)
        }
        topo.push(v.clone());
    }
}

impl Add for &Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Self::Output {
        // let back =
        // i want to define the backward function such that it sets self and rhs grad to be
        // out.grad * 1.0 for both self and rhs
        // the proof is c = a + b find dc/da
        // (a + h) + b - (a+b)/h
        // = h / h
        // so the backward function defined here is a field of the newly created node
        // but it mutates it's childrens grads
        let out = Value::from_prev(
            Value::data(self) + Value::data(rhs),
            vec![self.clone(), rhs.clone()],
            Op::Add,
        );

        // let back = |out: &Value| {
        //     let g = Value::grad(out);
        //     self.grad = 1.0 * g;
        //     rhs.grad = 1.0* g
        // }
        out
    }
}

impl Mul for &Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Self::Output {
        Value::from_prev(
            Value::data(self) * Value::data(rhs),
            vec![self.clone(), rhs.clone()],
            Op::Mul,
        )
    }
}

impl Neg for &Value {
    type Output = Value;
    fn neg(self) -> Self::Output {
        self * &Value::new(-1.0)
    }
}


fn main() -> io::Result<()> {
    let a = Value::new(3.0);
    let b = Value::new(-4.);

    let c = &a + &b;
    // println!("{:?}", c);
    // println!("{:?}, {:?}", Value::data(&c), Value::op(&c));
    let mut d = &a * &c;
    // let e = Value::tanh(&d);
    // println!("{:?}, {:?}", Value::data(&d), Value::op(&d));
    // e.backwards()
    // d.node.borrow_mut().grad = 1.0;
    // c.node.borrow_mut().grad += d.grad() * a.data();
    // a.node.borrow_mut().grad += d.grad() * c.data();

    // a.node.borrow_mut().grad += 1.0 * c.grad();
    // b.node.borrow_mut().grad += 1.0 * c.grad();

    println!("{:?}", d);
    println!("{:?}, {:?}", Value::data(&d), Value::op(&d));
    d.node.borrow_mut().grad = 1.0;
    // d.local_backwards();
    d.backwards();
    let graph_path = dot::render_graph(&d, "micrograd")?;
    println!("graph: {}", graph_path.display());

    // let x = Value::new(5.0);
    // let i1 = Value::new(2.);
    // let i2 = Value::new(3.);

    // let u = &x * &i1;
    // let v = &x * &i2;
    // let y = &u + &v;

    Ok(())
}

fn deez() {
    let h = 1e-4;
    let x = 3.;
    let f1 = f(x);
    let f2 = f(x + h);
    let df_dx = (f2 - f1) / h;
    println!("df_dx: {}", df_dx);

    let a = 2.;
    let b = -3.;
    let c = 10.;

    let d1 = a * b + c;
    let d2 = (a + h) * b + c;
    let dd_da = (d2 - d1) / h;

    println!("dd_da: {}", dd_da);
}
