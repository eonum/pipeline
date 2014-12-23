package ch.eonum.pipeline.classification.geneticlinearregression;

import java.util.Random;

import ch.eonum.pipeline.core.Features;

public class MinusNode extends BinaryNode {
	public MinusNode(double endNodesProb, double timesNodesProb,
			double thresholdNodesProb, Features features, Random random,
			int depth, int maxDepth) {
		super(endNodesProb, timesNodesProb, thresholdNodesProb, features,
				random, depth, maxDepth);
	}

	public MinusNode(Node left, Node right) {
		super(left, right);
	}

	@Override
	public String toString(){
		return "(" + left.toString() + " - " + right.toString() + ")";
	}
	
	@Override
	public double evaluate(double[] data) {
		return left.evaluate(data) - right.evaluate(data);
	}

	@Override
	public Node copy() {
		return new MinusNode(left.copy(), right.copy());
	}
	
	@Override
	public String getNodeName() {
		return "minusNode";
	}
}
