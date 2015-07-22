package ch.eonum.pipeline.classification.tree;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;

/**
 * Decision tree for classification.
 * 
 * @author tim
 *
 */
public class DecisionTreeClassifier<E extends Instance> extends DecisionTree<E> {

	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();

	static {
		PARAMETERS.put("u", "penalty for unequal splits (default 1.0)");
	}

	public DecisionTreeClassifier(Features features, int seed) {
		super(features, seed);
		this.getSupportedParameters().put("u", "penalty for unequal splits (default 1.0)");
		this.putParameter("u", 1.0);
	}

	public DecisionTreeClassifier(Features features, int seed,
			Map<String, List<Double>> splitsPerFeature, int i, Features classes) {
		super(features, seed, splitsPerFeature, i);
		this.classes = classes;
	}
	
	@Override
	protected SplitNode<E> createSplitNode() {
		return new SplitNodeClassifier<E>(this, 1, this.trainingDataSet, classes);
	}

	/**
	 * lossless pruning.
	 */
	public void prune() {
		this.getRoot().prune(new HashSet<Object>());
	}
}
