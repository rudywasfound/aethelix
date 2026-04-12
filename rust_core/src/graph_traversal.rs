use std::collections::HashMap;

pub struct CausalGraphState {
    // Maps a node to a list of its parents with associated edge weights
    parents_map: HashMap<String, Vec<(String, f64)>>,
}

impl CausalGraphState {
    pub fn new() -> Self {
        Self {
            parents_map: HashMap::new(),
        }
    }

    pub fn add_edge(&mut self, source: &str, target: &str, weight: f64) {
        self.parents_map
            .entry(target.to_string())
            .or_insert_with(Vec::new)
            .push((source.to_string(), weight));
    }

    pub fn get_parents(&self, node_name: &str) -> Vec<(String, f64)> {
        self.parents_map
            .get(node_name)
            .cloned()
            .unwrap_or_default()
    }

    /// Recursively find all paths from a sensor (observable) back to root causes,
    /// calculating the cumulative strength (product of weights) for each path.
    pub fn get_weighted_paths_to_root(
        &self, 
        node_name: &str, 
        max_depth: usize
    ) -> Vec<(Vec<String>, f64)> {
        if max_depth == 0 {
            return vec![];
        }

        let parents = self.get_parents(node_name);
        if parents.is_empty() {
            // This is a root cause
            return vec![(vec![node_name.to_string()], 1.0)];
        }

        let mut all_results = Vec::new();
        for (parent, weight) in parents {
            let parent_results = self.get_weighted_paths_to_root(&parent, max_depth - 1);
            for (mut path, parent_strength) in parent_results {
                path.push(node_name.to_string());
                all_results.push((path, parent_strength * weight));
            }
        }

        all_results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_traversal() {
        let mut graph = CausalGraphState::new();
        graph.add_edge("root1", "mid1", 0.5);
        graph.add_edge("mid1", "obs1", 0.8);
        graph.add_edge("root2", "obs1", 0.9);

        let paths = graph.get_weighted_paths_to_root("obs1", 10);
        assert_eq!(paths.len(), 2);
        
        // root1 -> mid1 -> obs1 (0.5 * 0.8 = 0.4)
        assert!(paths.contains(&(vec!["root1".to_string(), "mid1".to_string(), "obs1".to_string()], 0.4)));
        // root2 -> obs1 (0.9)
        assert!(paths.contains(&(vec!["root2".to_string(), "obs1".to_string()], 0.9)));
    }
}
