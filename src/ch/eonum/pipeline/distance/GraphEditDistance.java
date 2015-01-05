package ch.eonum.pipeline.distance;

import ch.eonum.pipeline.core.Graph;


/**
 * Calculate an edit distance between two graphs.
 * 
 * Not yet implemented.
 * 
 * @author tim
 *
 */
public class GraphEditDistance<E extends Graph> extends Distance<E> {
	
	/**
	 * Constructor. Initialize the zero graph.
	 * @param zeroGraph
	 */
	public GraphEditDistance(E zeroGraph) {
		super(zeroGraph);
	}

	public double distance(E graph1, E graph2) {
		// TODO Auto-generated method stub
		return 0;
	}

}
