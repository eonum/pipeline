package ch.eonum.pipeline.classification.geneticlinearregression;

import java.util.Random;

import ch.eonum.pipeline.core.Features;

public abstract class Node {

	public static Node randomNode(double endNodesProb, double timesNodesProb,
			double thresholdNodesProb, double nodeType, Features features, Random random, int depth, int maxDepth) {
		Node node = null;
		if (nodeType < endNodesProb || depth >= maxDepth){
			String feature = features.getFeatureByIndex(random
					.nextInt(features.size()));
			node = new EndNode(feature, features.getIndexFromFeature(feature));
		} else if (nodeType < timesNodesProb + endNodesProb)
			node = new TimesNode(endNodesProb, timesNodesProb, thresholdNodesProb, features,
					random, depth + 1, maxDepth);
		else if (nodeType < timesNodesProb + endNodesProb + thresholdNodesProb){
			String feature = features.getFeatureByIndex(random
					.nextInt(features.size()));
			node = new ThresholdNode(feature,
					features.getIndexFromFeature(feature), random.nextDouble());
		}
		else
			node = new MinusNode(endNodesProb, timesNodesProb, thresholdNodesProb, features,
					random, depth + 1, maxDepth);
		return node;
	}
	
	public abstract int getDepth();

	public abstract double evaluate(double[] data);

	public abstract Node copy();

	public abstract String getNodeName();

}
