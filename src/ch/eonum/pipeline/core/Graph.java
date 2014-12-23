package ch.eonum.pipeline.core;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Graph with nodes and edges. Nodes and edges can have attributes. Extends
 * SparseInstance. Hence the graph as a whole can have attributes.
 * 
 * Vector embedding (using graph edit distance to some prototypes) can be used
 * to fill up attributes in the sparse instance. Following vector embedding, all
 * standard classifiers and regressors can be used.
 * 
 * @author tim
 * 
 */
public class Graph extends SparseInstance {
	
	/**
	 * inner class for edges.
	 * @author tim
	 *
	 */
	public class Edge {
		public Node start;
		public Node end;
		public boolean directed;
		public Map<String, Double> attributes;
	}

	/**
	 * inner class for nodes.
	 * @author tim
	 *
	 */
	public class Node {
		public Map<String, Double> attributes;
	}

	/** nodes. **/
	private Set<Node> nodes;
	/** edges. **/
	private Set<Edge> edges;

	/**
	 * @param id
	 * @param gt
	 * @param vector
	 */
	public Graph(String id, String gt) {
		super(id, gt, new HashMap<String, Double>());
		this.nodes = new HashSet<Node>();
		this.edges = new HashSet<Edge>();
	}
	
	public Set<Node> nodes(){
		return nodes;
	}
	
	public Set<Edge> edges(){
		return edges;
	}
}
