package ch.eonum.pipeline.classification.geneticlinearregression;

import java.util.Random;

import ch.eonum.pipeline.core.Features;

/**
 * Abstract superclass of all nodes containing two nodes.
 * @author tim
 *
 */
public abstract class BinaryNode extends Node {
	/** left subtree. */
	protected Node left;
	/** right subtree. */
	protected Node right;
	
	public BinaryNode(Node left, Node right){
		this.left = left;
		this.right = right;
	}
	
	public BinaryNode(double endNodesProb, double timesNodesProb,
			double thresholdNodesProb, Features features, Random random, int depth, int maxDepth) {
		this.left = Node.randomNode(endNodesProb, timesNodesProb, thresholdNodesProb,
				random.nextDouble(), features, random, depth, maxDepth);
		this.right = Node.randomNode(endNodesProb, timesNodesProb, thresholdNodesProb,
				random.nextDouble(), features, random, depth, maxDepth);
	}
	
	@Override
	public int getDepth() {
		return Math.max(left.getDepth(), right.getDepth()) + 1;
	}
}
