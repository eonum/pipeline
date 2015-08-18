package ch.eonum.pipeline.classification.tree;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;

/**
 * Node in a decision tree for classification.
 * 
 * @author tim
 * 
 */
public class SplitNodeClassifier<E extends Instance> extends SplitNode<E> {

	/** class label. */
	private String label;
	/** classes. */
	private Features classes;
	private double u;

	public SplitNodeClassifier(DecisionTree<E> parent, int depth,
			DataSet<E> trainSet, Features classes) {
		super(parent, depth, trainSet);
		this.classes = classes;
		this.u = 1.;
	}

	@Override
	/**
	 * Select a split variable and value by maximizing the twoing measure.
	 */
	protected void pickSplitFeature(int minSize, Features fs) {
		double maxTwoing = Double.NEGATIVE_INFINITY;
		for (int f = 0; f < fs.size(); f++) {
			String feature = fs.getFeatureByIndex(f);
			List<Double> splits = parent.getSplitsForFeature(feature);
			for (Double sv : splits) {
				Map<String, Integer> geClassDistribution = createNullDistribution();
				Map<String, Integer> ltClassDistribution = createNullDistribution();
				int ltNum = 0;
				int geNum = 0;
				for (Instance each : trainSet)
					if (each.get(feature) >= sv) {
						geClassDistribution.put(each.groundTruth,
								geClassDistribution.get(each.groundTruth) + 1);
						geNum++;
					} else {
						ltClassDistribution.put(each.groundTruth,
								ltClassDistribution.get(each.groundTruth) + 1);
						ltNum++;
					}

				/** twoing splitting rule. */
				double q = ((double) geNum) / (geNum + ltNum);
				double twoing = Math.pow(q * (1 - q), u);
				double sum = 0.0;
				for (String className : classes.asSet()) {
					double pl = geNum == 0 ? 0.0 : geClassDistribution
							.get(className) / (double) geNum;
					double pr = ltNum == 0 ? 0.0 : ltClassDistribution
							.get(className) / (double) ltNum;
					sum += Math.abs(pl - pr);
				}
				twoing *= sum;

				if (twoing > maxTwoing && ltNum > minSize && geNum > minSize) {
					maxTwoing = twoing;
					splitFeature = feature;
					splitValue = sv;
				}
			}
		}
		if (maxTwoing == Double.NEGATIVE_INFINITY)
			return;
		DataSet<E> leftTrain = new DataSet<E>();
		DataSet<E> rightTrain = new DataSet<E>();
		for (E each : trainSet)
			if (each.get(splitFeature) >= splitValue) {
				leftTrain.add(each);
			} else {
				rightTrain.add(each);
			}
		left = new SplitNodeClassifier<E>(parent, depth + 1, leftTrain, classes);
		right = new SplitNodeClassifier<E>(parent, depth + 1, rightTrain,
				classes);
		left.train(); // #TODO parallelize
		right.train();
	}

	private Map<String, Integer> createNullDistribution() {
		Map<String, Integer> distribution = new HashMap<String, Integer>();
		for (String className : classes.asSet())
			distribution.put(className, 0);
		return distribution;
	}

	@Override
	protected void calculateValue() {
		Map<String, Integer> classDistribution = createNullDistribution();
		for (Instance each : trainSet)
			classDistribution.put(each.groundTruth,
					classDistribution.get(each.groundTruth) + 1);

		int maxValue = Integer.MIN_VALUE;
		for (String className : classes.asSet())
			if (classDistribution.get(className) > maxValue) {
				maxValue = classDistribution.get(className);
				label = className;
			}
	}

	@Override
	public void test(Instance each) {
		if (isTerminal())
			each.label = label;
		else {
			if (each.get(splitFeature) >= splitValue) {
				left.test(each);
			} else {
				right.test(each);
			}
		}
	}

	@Override
	public String toString() {
		String tree = "{ 'label' : " + label + ", 'num' : " + (trainSet == null ? "NaN" : trainSet.size());
		if (splitFeature != null)
			tree += ", 'splitOn' : '" + splitFeature + "', 'splitValue' : "
					+ splitValue;
		if (right != null)
			tree += ", \n'right' : " + indent(right.toString());
		if (left != null)
			tree += ", 'left' : " + indent(left.toString());
		tree += "}";
		return tree;
	}

	public String getLabel() {
		return this.label;
	}

	@Override
	public void prune(Set<Object> distinctLabels) {
		distinctLabels.add(this.label);

		Set<Object> below = new HashSet<Object>();
		if (this.left != null)
			this.left.prune(below);
		if (this.right != null)
			this.right.prune(below);

		if (below.size() == 1) {
			this.right = null;
			this.left = null;
		}

		distinctLabels.addAll(below);
	}
	
	public Map<String, Object> asMap() {
		Map<String, Object> node = super.asMap();
		node.put("label", this.label);
		return node;
	}
	
	public void loadFromJSON(DecisionTree<E> parent, Map<String, Object> json){
		super.loadFromJSON(parent, json);
		this.label = json.get("label").toString();
	}
}
