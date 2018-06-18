//! Recently, I've been trying to figure out a good way to model computation graphs in Rust. In this
//! post, I explore using a graph with vector indices. I'm not sure if this is the best approach,
//! but writing it out has helped me to understand the advantages and disadvantages better.
//!
//! When I say "computation graph," I mean a representation of a mathematical expression like
//! `2 * a + a * b`. This example contains a constant (`2`), two variables (`a` and `b`), and two
//! functions (addition and multiplication). This expression can be modeled as a directed acyclic
//! graph:
//!
//! ```text
//! 2   a   b
//!  \ / \ /
//!   *   *
//!    \ /
//!     +
//! ```
//!
//! In my ASCII diagram above, all edges go downwards. In general, edges don't really have any
//! interesting information associated with them so I'm going to pretty much ignore them.
//!
//! <!--more-->
//!
//! # A homogenous graph
//!
//! Below is a homogeneous DAG as a jumping-off point. This is strongly inspired by [Modeling graphs
//! in Rust using vector indices](http://smallcultfollowing.com/babysteps/blog/2015/04/06/modeling-graphs-in-rust-using-vector-indices/).
//! This isn't a computation graph, but it does let us implement an algorithm that traverses the
//! graph and memoizes intermediate values. This is important because the number of paths through a
//! DAG can grow exponentially with the number of nodes, meaning that recursive implementations
//! can be very slow.
//!
//! ```
//! /// Deriving Copy reduces ownership headaches
//! #[derive(Copy, Clone)]
//! pub struct Idx(usize);
//!
//! pub struct Node {
//!     children: Vec<Idx>,
//! }
//!
//! #[derive(Default)]
//! pub struct Graph {
//!     nodes: Vec<Node>,
//! }
//!
//! /// Graph maintains the invariant that nodes can only be added, never removed. This means that
//! /// a particular Idx will always be valid as long as it is used with the correct Graph.
//! impl Graph {
//!     pub fn push(&mut self, children: Vec<Idx>) -> Idx {
//!         self.nodes.push(Node { children });
//!         Idx(self.nodes.len() - 1)
//!     }
//!
//!     /// This returns the number of paths between each leaf node and the final node. This
//!     /// implementation memoizes the number of paths from leaves to each node.
//!     ///
//!     /// Note that "final node" is only a meaningful concept in a DAG where there is one node
//!     /// that is the ancestor of every other node in the graph; I'm using it here for simplicity.
//!     pub fn count_paths(&self) -> usize {
//!         let mut path_counts = Vec::new();
//!
//!         for node in &self.nodes {
//!             let paths_to_here = if node.children.is_empty() {
//!                 1
//!             } else {
//!                 node.children
//!                     .iter()
//!                     .map(|child_index| path_counts[child_index.0])
//!                     .sum()
//!             };
//!             path_counts.push(paths_to_here);
//!         }
//!
//!         path_counts[path_counts.len() - 1]
//!     }
//! }
//!
//! let mut g = Graph::default();
//! let a = g.push(vec![]);
//! let b = g.push(vec![a]);
//! let c = g.push(vec![a, b]);
//! let d = g.push(vec![a, b, c]);
//!
//! // All paths are:
//! // a -> b -> c -> d
//! // a -> b ------> d
//! // a ------> c -> d
//! // a -----------> d
//! assert_eq!(4, g.count_paths())
//! ```
//!
//! # What if Node is an enum?
//!
//! Next, a graph that can actually do some computation. I've installed a few upgrades relative to
//! the previous implementation:
//!
//! - There are three different kinds of `Node`: `Constant`, `Variable`, and `Sum`
//! - There is a `Subgraph` type that represents an ordered set of nodes
//! - `Idx` implements `std::ops::Add` for cute graph-building syntax
//! - There is a `derivative` method that transforms a subgraph
//! - `Graph` implements `Index<Idx>` for slightly more type safety
//!
//! ```
//! use std::collections::{HashMap, HashSet};
//! use std::ops::{Add, Index};
//!
//! /// To enable this to be used in HashMap and HashSet, this derives Eq, PartialEq, and Hash
//! #[derive(Copy, Clone, Eq, Hash, PartialEq)]
//! pub struct Idx(usize);
//!
//! impl Add for Idx {
//!     type Output = Node;
//!
//!     fn add(self, rhs: Idx) -> Node {
//!         Node::Sum { children: vec![self, rhs] }
//!     }
//! }
//!
//! pub enum Node {
//!     Constant(f64),
//!     Variable,
//!     Sum { children: Vec<Idx> },
//! }
//!
//! impl Node {
//!     fn get_value(&self, my_index: &Idx, values: &HashMap<Idx, f64>) -> f64 {
//!         match self {
//!             Node::Constant(value) => *value,
//!             Node::Variable => values[my_index],
//!             Node::Sum { children } => children.iter().map(|child| values[child]).sum(),
//!         }
//!     }
//!
//!     fn derivative(
//!         &self,
//!         my_index: &Idx,
//!         wrt: &HashSet<Idx>,
//!         derivatives: &HashMap<Idx, Idx>,
//!     ) -> Node {
//!         match self {
//!             Node::Constant(_) => Node::Constant(0.0),
//!             Node::Variable => {
//!                 if wrt.contains(my_index) {
//!                     Node::Constant(1.0)
//!                 } else {
//!                     Node::Constant(0.0)
//!                 }
//!             }
//!             Node::Sum { ref children } => {
//!                 Node::Sum {
//!                     children: children.iter().map(|child| derivatives[child]).collect(),
//!                 }
//!             }
//!         }
//!     }
//! }
//!
//! /// This helps us to represent the idea that only a subset of the nodes in a graph might be
//! /// relevant for a particular computation. The indices in a Subgraph are ordered such that a
//! /// child always comes before one of its parents.
//! pub struct Subgraph {
//!     indices: Vec<Idx>,
//! }
//!
//! impl Subgraph {
//!     fn new(indices_unsorted: impl Iterator<Item = Idx>) -> Self {
//!         let mut indices: Vec<Idx> = indices_unsorted.collect();
//!
//!         // This is an easy way to enforce the order condition
//!         indices.sort_unstable_by_key(|index| index.0);
//!         Self { indices: indices }
//!     }
//! }
//!
//! #[derive(Default)]
//! pub struct Graph {
//!     nodes: Vec<Node>,
//! }
//!
//! impl Graph {
//!     pub fn push(&mut self, node: Node) -> Idx {
//!         self.nodes.push(node);
//!         Idx(self.nodes.len() - 1)
//!     }
//!
//!     pub fn as_subgraph(&self) -> Subgraph {
//!         Subgraph { indices: self.nodes.iter().enumerate().map(|(i, _)| Idx(i)).collect() }
//!     }
//!
//!     /// Given values for each relevant variable, this computes the value for each node in the
//!     /// graph.
//!     pub fn evaluate_subgraph(
//!         &self,
//!         subgraph: Subgraph,
//!         variable_to_value: HashMap<Idx, f64>,
//!     ) -> HashMap<Idx, f64> {
//!         let mut result = variable_to_value;
//!
//!         for index in subgraph.indices.iter() {
//!             let value = self[*index].get_value(index, &result);
//!             result.insert(*index, value);
//!         }
//!
//!         result
//!     }
//!
//!     pub fn evaluate(&self, variable_to_value: HashMap<Idx, f64>) -> HashMap<Idx, f64> {
//!         self.evaluate_subgraph(self.as_subgraph(), variable_to_value)
//!     }
//!
//!     /// This transforms the graph by taking the derivative
//!     pub fn derivative(&mut self, of: Idx, wrt: HashSet<Idx>) -> (Idx, Subgraph) {
//!         // Memoize the derivative of each node
//!         let mut derivatives: HashMap<Idx, Idx> = HashMap::new();
//!
//!         for old_index in 0..self.nodes.len() {
//!             let old_index = Idx(old_index);
//!             let new_node = self[old_index].derivative(&old_index, &wrt, &derivatives);
//!             let new_index = self.push(new_node);
//!             derivatives.insert(old_index, new_index);
//!         }
//!
//!         // The subgraph contains all the new nodes we just created
//!         (
//!             derivatives[&of],
//!             Subgraph::new(derivatives.values().cloned()),
//!         )
//!     }
//! }
//!
//! impl Index<Idx> for Graph {
//!     type Output = Node;
//!
//!     fn index(&self, index: Idx) -> &Node {
//!         &self.nodes[index.0]
//!     }
//! }
//!
//! // c = 1 + b
//! let mut g = Graph::default();
//! let a = g.push(Node::Constant(1.0));
//! let b = g.push(Node::Variable);
//! let c = g.push(a + b);
//!
//! // 1 + 2 = 3
//! let variable_to_value = {
//!     let mut result = HashMap::new();
//!     result.insert(b, 2.0);
//!     result
//! };
//! assert_eq!(3.0, g.evaluate(variable_to_value)[&c]);
//!
//! // The derivative of c wrt b is just 1
//! let wrt = {
//!     let mut result = HashSet::new();
//!     result.insert(b);
//!     result
//! };
//! let (d_c_b, subgraph) = g.derivative(c, wrt);
//! assert_eq!(1.0, g.evaluate_subgraph(subgraph, HashMap::new())[&d_c_b]);
//! ```
//!
//! Overall, I'm pretty happy with this implementation. However, adding many different types of node
//! will cause the match statements to balloon. Additionally, I would rather see all the code for
//! one type of `Node` in one place.
//!
//! # What if Node is a trait?
//! To solve this, I'm going to make `Node` a trait instead of an enum. I have hidden aspects of the
//! implementation that are the same as in the previous implementation; the complete implementation
//! is in the source code for this post.
//!
//! ```
//! # use std::collections::{HashMap, HashSet};
//! # use std::ops::{Add, Index};
//! # #[derive(Copy, Clone, Eq, Hash, PartialEq)]
//! # pub struct Idx(usize);
//! # impl Add for Idx {
//! #    type Output = Box<Node>;
//! #    fn add(self, rhs: Idx) -> Box<Node> {
//! #        Box::from(Sum { children: vec![self, rhs] })
//! #    }
//! # }
//! pub trait Node: 'static {
//!     /// The input must include values for all variables and for all children of this node.
//!     fn get_value(&self, my_index: &Idx, values: &HashMap<Idx, f64>) -> f64;
//!
//!     fn derivative(
//!         &self,
//!         my_index: &Idx,
//!         wrt: &HashSet<Idx>,
//!         derivatives: &HashMap<Idx, Idx>,
//!     ) -> Box<Node>;
//! }
//!
//! pub struct Constant(f64);
//!
//! impl Node for Constant {
//!     fn get_value(&self, _my_index: &Idx, _values: &HashMap<Idx, f64>) -> f64 {
//!         self.0
//!     }
//!
//!     fn derivative(
//!         &self,
//!         _my_index: &Idx,
//!         _wrt: &HashSet<Idx>,
//!         _derivatives: &HashMap<Idx, Idx>,
//!     ) -> Box<Node> {
//!         Box::from(Constant(0.0))
//!     }
//! }
//!
//! pub struct Variable;
//!
//! impl Node for Variable {
//!     fn get_value(&self, _my_index: &Idx, _values: &HashMap<Idx, f64>) -> f64 {
//!         _values[_my_index]
//!     }
//!
//!     fn derivative(
//!         &self,
//!         my_index: &Idx,
//!         wrt: &HashSet<Idx>,
//!         _derivatives: &HashMap<Idx, Idx>,
//!     ) -> Box<Node> {
//!         if wrt.contains(my_index) {
//!             Box::from(Constant(1.0))
//!         } else {
//!             Box::from(Constant(0.0))
//!         }
//!     }
//! }
//!
//! pub struct Sum {
//!     children: Vec<Idx>,
//! }
//!
//! impl Node for Sum {
//!     fn get_value(&self, _my_index: &Idx, _values: &HashMap<Idx, f64>) -> f64 {
//!         self.children.iter().map(|child| _values[child]).sum()
//!     }
//!
//!     fn derivative(
//!         &self,
//!         _my_index: &Idx,
//!         _wrt: &HashSet<Idx>,
//!         derivatives: &HashMap<Idx, Idx>,
//!     ) -> Box<Node> {
//!         Box::from(Sum {
//!             children: self.children
//!                 .iter()
//!                 .map(|child| derivatives[child])
//!                 .collect(),
//!         })
//!     }
//! }
//!
//! # pub struct Subgraph {
//! #     indices: Vec<Idx>,
//! # }
//! # impl Subgraph {
//! #    fn new(indices_unsorted: impl Iterator<Item = Idx>) -> Self {
//! #        let mut indices: Vec<Idx> = indices_unsorted.collect();
//! #        indices.sort_unstable_by_key(|index| index.0);
//! #        Self { indices }
//! #    }
//! # }
//! /// Since Node does not implement Sized, we need to box it so we can put it into a Vec.
//! #[derive(Default)]
//! pub struct Graph {
//!     nodes: Vec<Box<Node>>,
//! }
//!
//! /// This is almost identical to the Graph implementation with the enum, except that the push
//! /// fn now accepts a Box<Node>, and I've added push_box.
//! impl Graph {
//!     pub fn push_box(&mut self, box_node: Box<Node>) -> Idx {
//!         self.nodes.push(box_node);
//!         Idx(self.nodes.len() - 1)
//!     }
//!
//!     pub fn push<N: Node>(&mut self, node: N) -> Idx {
//!         self.push_box(Box::from(node))
//!     }
//! #   pub fn as_subgraph(&self) -> Subgraph {
//! #       Subgraph { indices: self.nodes.iter().enumerate().map(|(i, _)| Idx(i)).collect() }
//! #   }
//! #   pub fn evaluate_subgraph(
//! #       &self,
//! #       subgraph: Subgraph,
//! #       variable_to_value: HashMap<Idx, f64>,
//! #    ) -> HashMap<Idx, f64> {
//! #       let mut result = variable_to_value;
//! #       for index in subgraph.indices.iter() {
//! #           let value = self[*index].get_value(index, &result);
//! #           result.insert(*index, value);
//! #       }
//! #        result
//! #   }
//! #   pub fn evaluate(&self, variable_to_value: HashMap<Idx, f64>) -> HashMap<Idx, f64> {
//! #       self.evaluate_subgraph(self.as_subgraph(), variable_to_value)
//! #   }
//! #   pub fn derivative(&mut self, of: Idx, wrt: HashSet<Idx>) -> (Idx, Subgraph) {
//! #       let mut derivatives: HashMap<Idx, Idx> = HashMap::new();
//! #       for old_index in 0..self.nodes.len() {
//! #           let old_index = Idx(old_index);
//! #           let new_node = self[old_index].derivative(&old_index, &wrt, &derivatives);
//! #           let new_index = self.push_box(new_node);
//! #           derivatives.insert(old_index, new_index);
//! #       }
//! #       (
//! #           derivatives[&of],
//! #           Subgraph::new(derivatives.values().cloned()),
//! #       )
//! #   }
//! }
//!
//! # impl Index<Idx> for Graph {
//! #     type Output = Node;
//! #     fn index(&self, index: Idx) -> &Node {
//! #         &*self.nodes[index.0]
//! #     }
//! # }
//! // c = 1 + b
//! let mut g = Graph::default();
//! let a = g.push(Constant(1.0));
//! let b = g.push(Variable);
//! let c = g.push_box(a + b);
//!
//! // 1 + 2 = 3
//! let variable_to_value = {
//!     let mut result = HashMap::new();
//!     result.insert(b, 2.0);
//!     result
//! };
//! assert_eq!(3.0, g.evaluate(variable_to_value)[&c]);
//!
//! // The derivative of c wrt b is just 1
//! let wrt = {
//!     let mut result = HashSet::new();
//!     result.insert(b);
//!     result
//! };
//! let (d_c_b, subgraph) = g.derivative(c, wrt);
//! assert_eq!(1.0, g.evaluate_subgraph(subgraph, HashMap::new())[&d_c_b]);
//!```
//!
//! While implementing this, I noticed that making `Node` a trait enforces a clean separation of
//! responsibility between the graph and the node. I actually used the implementation of this
//! version to clean up the factorization of the enum version.
//!
//! However, making `Node` a trait brings with it the ergonomic disadvantage that nodes often need
//! to be passed around inside a `Box`, which is slightly annoying. Separately, there is a
//! performance penalty because we are now using dynamic dispatch instead of static dispatch. I
//! don't think that I care too much about this, because I'm interested in using these graphs with
//! large tensors where the cost of the actual computation will dwarf the cost of traversing the
//! graph.
//!
//! # Advantages
//!
//! I was happy about several aspects of this experiment:
//!
//! - I did not need to introduce explicit lifetimes at all.
//! - I think it will be possible to construct one of these graphs at runtime. This means that a
//!   serialized graph could be loaded in from a file, for example.
//! - I didn't need to rely on any external dependencies. This is not usually a goal of mine but it
//!   makes the code easier to understand.
//! - Having `Idx` implement `Copy` makes it more pleasant to work with.
//!
//! # Disadvantages
//!
//! - The syntax for creating a graph is a bit cumbersome, since we have to write `g.push(...)`
//!   every time we want to add a node to the graph.
//! - Having `Graph`, `Subgraph`, `Node`, and `Idx` is a lot of structs even for this toy
//!   implementation.
//! - I don't entirely understand why `Node` needs to have the static lifetime. Hopefully that
//!   doesn't mean anything bad.
//! - Since a `Node` doesn't know its own index, the index needs to be passed around a lot.
//!
//! # Future directions
//!
//! I have an unhealthy obsession with building an elegant DSL in vanilla Rust. I would love to be
//! able to create a graph by writing something like this:
//!
//! ```text
//! let x = variable();
//! let y = variable();
//! let z = x * 2.0 + y;
//! ```
//!
//! One somewhat crazy direction of exploration would be to allow different nodes in the graph to
//! implement different traits.
//!
//! I would like to be able to save and load graphs.
//!
//! It would be good to have a way to represent functions that are composed of smaller functions,
//! like softmax.
//!
//! These questions aside, the most obvious ways to make this more useful would be to implement many
//! different functions and to allow computation on data such as tensors.
//!
//! # About
//!
//! This blog post was produced using [cargo-readme](https://docs.rs/cargo-readme) to ensure that
//! all of the code actually works. The source code is [here](https://github.com/paulkernfeld/exploring-computation-graphs-in-rust).
