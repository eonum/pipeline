package ch.eonum.pipeline.classification.geneticlinearregression;

import java.util.Random;

import ch.eonum.pipeline.core.Features;

public class TimesNode extends BinaryNode {
	public TimesNode(double endNodesProb, double timesNodesProb,
			double thresholdNodesProb, Features features, Random random,
			int depth, int maxDepth) {
		super(endNodesProb, timesNodesProb, thresholdNodesProb, features,
				random, depth, maxDepth);
	}

	public TimesNode(Node left, Node right) {
		super(left, right);
	}

	@Override
	public String toString(){
		return "(" + left.toString() + " * " + right.toString() + ")";
	}

	@Override
	public double evaluate(double[] data) {
		return left.evaluate(data) * right.evaluate(data);
	}
	
	@Override
	public Node copy() {
		return new TimesNode(left.copy(), right.copy());
	}
	
	@Override
	public String getNodeName() {
		return "timesNode";
	}
}
