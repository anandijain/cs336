mod dot;

use std::{
    cell::RefCell,
    collections::HashSet,
    ops::{Add, Mul, Sub},
    rc::Rc,
};

#[derive(Debug, Clone)]
struct Val {
    node: Rc<RefCell<Node>>,
}

impl Val {
    fn new(data: f64) -> Self {
        Val::from_prev(data, vec![], Op::Leaf)
    }

    fn from_prev(data: f64, prev: Vec<Val>, op: Op) -> Self {
        Val {
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
    fn set_grad(&self, g: f64) {
        self.node.borrow_mut().grad = g
    }
    fn prev(&self) -> Vec<Val> {
        self.node.borrow().prev.clone()
    }

    // assume that self.grad() is correct and then apply the chain rule for prev
    fn local_backwards(&self) {
        let d = self.data();
        let g = self.grad();
        let prev = self.node.borrow_mut().prev.clone();
        match self.op() {
            Op::Leaf => {}
            Op::Add => {
                prev.iter().for_each(|v| v.set_grad(v.grad() + g * 1.0));
            }
            Op::Mul => {
                assert_eq!(prev.len(), 2);
                let lhs = prev[0].clone();
                let rhs = prev[1].clone();
                lhs.set_grad(lhs.grad() + g * rhs.data());
                rhs.set_grad(rhs.grad() + g * lhs.data());
            }
            Op::Tanh => {
                let chd = prev[0].clone();
                chd.set_grad(chd.grad() + g * (1.0 - d.powi(2)));
            }
            Op::Pow(exp) => {
                // self.data = d = x^n
                //
                // d =
                let chd = prev[0].clone();
                let local_grad = exp as f64 * chd.data().powi(exp - 1);
                chd.set_grad(chd.grad() + g * local_grad);
            }
        }
    }

    fn backward(&self) {
        self.set_grad(1.0);
        let mut topo = build_topo(self);
        topo.reverse();
        topo.iter().for_each(|v| v.local_backwards());
    }

    fn tanh(&self) -> Self {
        Self::from_prev(self.data().tanh(), vec![self.clone()], Op::Tanh)
    }

    fn pow(&self, exp: i32) -> Self {
        Self::from_prev(self.data().powi(exp), vec![self.clone()], Op::Pow(exp))
    }
}

impl Add for &Val {
    type Output = Val;
    fn add(self, rhs: Self) -> Self::Output {
        Val::from_prev(
            self.data() + rhs.data(),
            vec![self.clone(), rhs.clone()],
            Op::Add,
        )
    }
}

impl Sub for &Val {
    type Output = Val;
    fn sub(self, rhs: Self) -> Self::Output {
        self + &(&Val::new(-1.0) * rhs)
    }
}

impl Mul for &Val {
    type Output = Val;
    fn mul(self, rhs: Self) -> Self::Output {
        Val::from_prev(
            self.data() * rhs.data(),
            vec![self.clone(), rhs.clone()],
            Op::Mul,
        )
    }
}

#[derive(Debug, Clone)]
struct Node {
    data: f64,
    grad: f64,
    prev: Vec<Val>,
    op: Op,
}

#[derive(Debug, Copy, Clone)]
enum Op {
    Leaf,
    Add,
    Mul,
    Tanh,
    Pow(i32),
}

impl Op {
    fn symbol(self) -> String {
        match self {
            Op::Leaf => "".to_string(),
            Op::Add => "+".to_string(),
            Op::Mul => "*".to_string(),
            Op::Tanh => "tanh".to_string(),
            Op::Pow(x) => format!("pow({x}"),
        }
    }
}

/// returns a topogically sorted list of the values of the DAG
fn build_topo(v: &Val) -> Vec<Val> {
    let mut visited = HashSet::new();
    let mut topo = vec![];
    visit(v, &mut visited, &mut topo);
    topo
}

fn visit(v: &Val, visited: &mut HashSet<usize>, topo: &mut Vec<Val>) {
    let ptr_v = v.node.as_ptr() as usize;
    if !visited.contains(&ptr_v) {
        visited.insert(ptr_v);
        v.prev().iter().for_each(|val| visit(val, visited, topo));
        topo.push(v.clone());
    }
}

// perceptron function is that it has a weights vector that weights its imputs
// tanh(dot(x, self.w) + self.b)
struct Neuron {
    w: Vec<Val>,
    b: Val,
}

impl Neuron {
    fn new(n_in: u64) -> Self {
        Neuron {
            w: (0..n_in)
                .map(|_| Val::new(rand::random_range(-1. ..1.)))
                .collect(),
            b: Val::new(rand::random_range(-1. ..1.)),
        }
    }
}

impl Module for Neuron {
    fn forward(&self, x: Vec<Val>) -> Vec<Val> {
        // assert_eq!(self.w.len(), x.len());
        // let mut res = self.b.clone();
        // for (wi, xi) in self.w.iter().zip(x) {
        //     res = &res + &(wi * &xi);
        // }
        // vec![res.tanh()]

        // vec![
        let res = self
            .w
            .iter()
            .zip(x)
            // .map(|(wi, xi)| wi * &xi)
            .fold(self.b.clone(), |sum, (wi, xi)| &sum + &(wi * &xi))
            .tanh();
        // ]
        vec![res]
    }

    fn parameters(&self) -> Vec<Val> {
        // self.w.push(self.b
        let mut params = self.w.clone();
        params.push(self.b.clone());
        params
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new(n_in: u64, n_out: u64) -> Self {
        Layer {
            neurons: (0..n_out).map(|_| Neuron::new(n_in)).collect(),
        }
    }
}

impl Module for Layer {
    fn forward(&self, x: Vec<Val>) -> Vec<Val> {
        self.neurons
            .iter()
            .flat_map(|n| n.forward(x.clone()))
            .collect()
    }

    fn parameters(&self) -> Vec<Val> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    fn new(n_in: u64, mut n_outs: Vec<u64>) -> Self {
        n_outs.insert(0, n_in);
        MLP {
            layers: (0..n_outs.len() - 1)
                .map(|idx| Layer::new(n_outs[idx], n_outs[idx + 1]))
                .collect(),
        }
    }
}
impl Module for MLP {
    fn forward(&self, x: Vec<Val>) -> Vec<Val> {
        let mut o = x.clone();
        for l in &self.layers {
            o = l.forward(o)
        }
        o
    }

    fn parameters(&self) -> Vec<Val> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

trait Module {
    fn forward(&self, x: Vec<Val>) -> Vec<Val>;
    fn parameters(&self) -> Vec<Val>;

    fn zero_grad(&self) {
        self.parameters().iter().for_each(|p| p.set_grad(0.0));
    }
}

fn values(xs: impl IntoIterator<Item = impl Into<f64>>) -> Vec<Val> {
    xs.into_iter().map(|x| Val::new(x.into())).collect()
}

fn main() {
    // let xs = values([1.0, 2.0, -3.0, -4.0]);
    // let mlp = MLP::new(3, vec![4, 4, 1]);
    // let out = mlp.forward(xs);
    // assert_eq!(out.len(), 1);
    // let o = out[0].clone();
    // o.backward();

    let mlp = MLP::new(3, vec![4, 4, 1]);
    println!("mlp parameters len: {}", mlp.parameters().len());

    let xs = [
        values([2.0, 3.0, -1.0]),
        values([3.0, -1.0, 0.5]),
        values([0.5, 1.0, 1.0]),
        values([1.0, 1.0, -1.0]),
    ];

    let ys = values([1.0, -1.0, -1.0, 1.0]);
    let lr = 1e-2;

    for i in (0..100) {
        let ypred = xs.clone().map(|x| mlp.forward(x)[0].clone());

        let loss = ys
            .iter()
            .zip(ypred)
            .fold(Val::new(0.0), |sum, (yi, ypi)| &sum + &(yi - &ypi).pow(2));
        println!("loss{i} = {}", loss.data());
        loss.backward();

        mlp.parameters()
            .iter()
            .for_each(|p| p.node.borrow_mut().data += -lr * p.grad());
        mlp.zero_grad();
    }
}
