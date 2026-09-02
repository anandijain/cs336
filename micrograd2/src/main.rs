mod dot;

use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::{Debug, Display},
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

use rand::random_range;

#[derive(Clone)]
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

impl Value {
    fn new(data: f64) -> Self {
        Value::with_prev(data, vec![], Op::Leaf)
    }

    fn with_prev(data: f64, prev: Vec<Value>, op: Op) -> Self {
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

    // fn children(&self) -> Vec<Value> {
    //     self.node.borrow().prev.clone()
    // }

    fn set_grad(&self, g: f64) {
        self.node.borrow_mut().grad = g;
    }

    /// we assume that self has it's gradient set
    fn local_backwards(&self) {
        let g = self.grad();
        let chds = self.node.borrow().prev.clone();

        match self.op() {
            Op::Leaf => {}
            Op::Add => {
                chds.iter()
                    .for_each(|n| n.node.borrow_mut().grad += 1.0 * g);
            }
            Op::Mul => {
                let [lhs, rhs]: [Value; 2] = chds.try_into().unwrap();
                let lx = lhs.data();
                let rx = rhs.data();
                lhs.node.borrow_mut().grad += g * rx;
                rhs.node.borrow_mut().grad += g * lx;
            }
            Op::Tanh => {
                let d = 1. - self.data().powi(2);
                chds[0].node.borrow_mut().grad += g * d;
            }
            Op::Exp => chds[0].node.borrow_mut().grad += g * self.data(), // _ => todo!()
            Op::Pow(exp) => {
                let base = chds[0].data();
                let local_der = (exp as f64 * base.powi(exp - 1));
                chds[0].node.borrow_mut().grad += g * local_der;
            }
        }
    }

    fn backward(&self) {
        self.set_grad(1.0);
        let topo = build_topo(self.clone());
        topo.iter().for_each(|v| v.local_backwards());
    }

    fn tanh(&self) -> Value {
        // let x = self.data().tanh();
        let o = (2. * self).exp();
        let x = &(&o - 1.) / &(&o + 1.);
        x
        // Value::with_prev(x, vec![self.clone()], Op::Tanh)
    }

    fn exp(&self) -> Value {
        Value::with_prev(self.data().exp(), vec![self.clone()], Op::Exp)
    }

    fn pow(&self, exp: i32) -> Value {
        Value::with_prev(self.data().powi(exp), vec![self.clone()], Op::Pow(exp))
    }
}

impl Add for &Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Self::Output {
        Value::with_prev(
            self.data() + rhs.data(),
            vec![self.clone(), rhs.clone()],
            Op::Add,
        )
    }
}

impl Add<f64> for &Value {
    type Output = Value;
    fn add(self, rhs: f64) -> Self::Output {
        self + &Value::new(rhs)
    }
}

impl Add<&Value> for f64 {
    type Output = Value;
    fn add(self, rhs: &Value) -> Self::Output {
        rhs + self
    }
}

impl Sub for &Value {
    type Output = Value;
    fn sub(self, rhs: Self) -> Self::Output {
        &-rhs + self
    }
}

impl Sub<f64> for &Value {
    type Output = Value;
    fn sub(self, rhs: f64) -> Self::Output {
        self - &Value::new(rhs)
    }
}

impl Sub<&Value> for f64 {
    type Output = Value;
    fn sub(self, rhs: &Value) -> Self::Output {
        &Value::new(self) - rhs
    }
}

impl Mul for &Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Self::Output {
        Value::with_prev(
            self.data() * rhs.data(),
            vec![self.clone(), rhs.clone()],
            Op::Mul,
        )
    }
}

impl Mul<f64> for &Value {
    type Output = Value;
    fn mul(self, rhs: f64) -> Self::Output {
        self * &Value::new(rhs)
    }
}

impl Mul<&Value> for f64 {
    type Output = Value;
    fn mul(self, rhs: &Value) -> Self::Output {
        rhs * self
    }
}

impl Neg for &Value {
    type Output = Value;
    fn neg(self) -> Self::Output {
        &Value::new(-1.) * self
    }
}

impl Div for &Value {
    type Output = Value;
    fn div(self, rhs: Self) -> Self::Output {
        self * &rhs.pow(-1)
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(data={:.4}, grad={:.4})", self.data(), self.grad())
    }
}

fn values<T: Into<f64>>(items: impl IntoIterator<Item = T>) -> Vec<Value> {
    items
        .into_iter()
        .map(|item| Value::new(item.into()))
        .collect()
}

#[derive(Debug, Copy, Clone)]
enum Op {
    Leaf,
    Add,
    Mul,
    Tanh,
    Exp,
    Pow(i32),
}

impl Op {
    fn symbol(self) -> String {
        match self {
            Op::Leaf => "".into(),
            Op::Add => "+".into(),
            Op::Mul => "*".into(),
            Op::Tanh => "tanh".into(),
            Op::Exp => "exp".into(),
            Op::Pow(x) => format!("pow{}", x),
        }
    }
}

fn build_topo(v: Value) -> Vec<Value> {
    let mut topo = vec![];
    let mut visited = HashSet::new();
    visit(v, &mut visited, &mut topo);
    topo.reverse();
    topo
}

fn visit(v: Value, visited: &mut HashSet<usize>, topo: &mut Vec<Value>) {
    let vptr = v.node.as_ptr() as usize;

    if !visited.contains(&vptr) {
        visited.insert(vptr);
        v.node
            .borrow()
            .prev
            .iter()
            .for_each(|chd| visit(chd.clone(), visited, topo));
        topo.push(v)
    }
}

#[derive(Debug, Clone)]
struct Neuron {
    w: Vec<Value>,
    b: Value,
}

impl Neuron {
    fn new(n_in: usize) -> Self {
        Neuron {
            w: (0..n_in)
                .map(|_| Value::new(random_range(-1. ..1.)))
                .collect(),
            b: Value::new(random_range(-1. ..1.)),
        }
    }
}

trait Module {
    fn forward(&self, inputs: &[Value]) -> Vec<Value>;

    fn parameters(&self) -> Vec<Value>;

    fn zero_grad(&self) {
        self.parameters().iter().for_each(|p| p.set_grad(0.));
    }
}

impl Module for Neuron {
    fn forward(&self, inputs: &[Value]) -> Vec<Value> {
        assert_eq!(inputs.len(), self.w.len());

        vec![
            self.w
                .iter()
                .zip(inputs)
                .map(|(wi, xi)| wi * xi)
                .fold(self.b.clone(), |sum, product| &sum + &product)
                .tanh(),
        ]
    }

    fn parameters(&self) -> Vec<Value> {
        let mut params = self.w.clone();
        params.push(self.b.clone());
        params
    }
}

#[derive(Debug, Clone)]
struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new(n_in: usize, n_out: usize) -> Self {
        let neurons = (0..n_out).map(|_| Neuron::new(n_in)).collect();
        Layer { neurons }
    }
}

impl Module for Layer {
    fn forward(&self, inputs: &[Value]) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|n| n.forward(inputs)[0].clone())
            .collect()
    }

    fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

#[derive(Debug, Clone)]
struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    fn new(n_in: usize, n_outs: &mut Vec<usize>) -> Self {
        n_outs.insert(0, n_in);
        let layers = (0..(n_outs.len() - 1))
            .map(|idx| Layer::new(n_outs[idx], n_outs[idx + 1]))
            .collect();
        MLP { layers }
    }
}

impl Module for MLP {
    fn forward(&self, inputs: &[Value]) -> Vec<Value> {
        let mut o1 = inputs.to_vec();
        for l in &self.layers {
            o1 = l.forward(&o1);
        }
        o1
    }

    fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|n| n.parameters()).collect()
    }
}

fn main() {
    // let mlp = MLP::new(3, &mut vec![4, 4, 1]);
    // println!("3,4,4,1 mlp params {:?}", mlp.parameters().len());

    // let out = mlp.forward(&[Value::new(2.), Value::new(3.), Value::new(4.)])[0].clone();
    // println!("{:?}", out);

    // let xs = [
    //     values([2.0, 3.0, -1.0]),
    //     values([3.0, -1.0, 0.5]),
    //     values([0.5, 1.0, 1.0]),
    //     values([1.0, 1.0, -1.0]),
    // ];

    // let ys = [1.0, -1.0, -1.0, 1.0].to_vec();

    // for i in 0..100 {
    //     let ypred = xs.clone().map(|x| {
    //         mlp.forward(&x)
    //             .into_iter()
    //             .next()
    //             .expect("MLP should produce one output")
    //     });

    //     let loss = ys
    //         .iter()
    //         .zip(ypred)
    //         .fold(Value::new(0.), |acc, (y, yhat)| &acc + &(*y - &yhat).pow(2));
    //     loss.backward();

    //     println!("loss {i}: {:?}", loss);
    //     let lr = 1e-1;
    //     mlp.parameters()
    //         .iter()
    //         .for_each(|p| p.node.borrow_mut().data += -lr * p.grad());
    //     mlp.zero_grad();
    // }
    // let ypred = xs.clone().map(|x| {
    //     mlp.forward(&x)
    //         .into_iter()
    //         .next()
    //         .expect("MLP should produce one output")
    // });

    // println!("{ypred:?}");

    let a = Value::new(3.5);
    let b = Value::new(-2.0);
    let c = &a * &b; // c = a * b
    let d = &c + &a;
    d.backward();
    println!("a {a:?}\nb {b:?}\nc {c:?}\nd {d:?}\n");

    // let graph_path = dot::render_graph(&loss, "micrograd").unwrap();
    // println!("graph: {}", graph_path.display());
}
