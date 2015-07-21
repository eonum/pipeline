package ch.eonum.pipeline.classification.tree;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.util.Log;

/**
 * CART decision tree for regression. @link http://en.wikipedia.org/wiki/Decision_tree_learning
 * @author tim
 *
 */
public class DecisionTree<E extends Instance> extends Classifier<E> implements Runnable {
	
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {
		PARAMETERS.put("k", "percentage of features which are taken into account at each split node [0,1] (default 1.0)");
		PARAMETERS.put("maxDepth", "maximum depth of the tree (default 100.0)");
		PARAMETERS.put("minSize", "minimum number of training instances within one terminal node (default 5.0)");
		PARAMETERS.put("tolerance", "tolerance range for distinct values (default 0.01)");
	}
	
	/** root node for the tree. */
	private SplitNode<E> root;
	private Random rand;
	private Map<String, List<Double>> splitsPerFeature;
	/** tree number when used in an ensemble (random forest). */
	private int treeNumber;


	public DecisionTree(Features features, int seed) {
		this.setFeatures(features);
		this.setSupportedParameters(DecisionTree.PARAMETERS);
		this.putParameter("k", 1.0);
		this.putParameter("maxDepth", 100.0);
		this.putParameter("minSize", 100.0);
		this.putParameter("tolerance", 0.01);
		this.rand = new Random(seed);
		treeNumber = -1;
	}

	public DecisionTree(Features features, int seed,
			Map<String, List<Double>> spf, int treeNumber) {
		this(features, seed);
		this.splitsPerFeature = spf;
		this.treeNumber = treeNumber;
	}

	@Override
	public void train() {
		if(this.splitsPerFeature == null)
			splitsPerFeature = calculateSplitsPerFeature(features,
					trainingDataSet, this.getDoubleParameter("tolerance"));
		
		if(this.classes == null){
			this.classes = new Features();
			Set<String> classes = trainingDataSet.collectClasses();
			for(String className : classes)
				this.classes.addFeature(className);
			this.classes.recalculateIndex();
		}
		this.root = createSplitNode();
		root.train();
		
		Log.puts(((treeNumber != -1) ? "== Tree " + treeNumber + " ==\n": "") + root);
	}
	
	@Override
	public Map<String, Object> asMap(){
		Map<String, Object> dbo = super.asMap();
		dbo.put("classify", classify);
		List<String> list;
		if(classes != null){
			list = classes.asStringList();
			dbo.put("classes", list);
		}
		if(features != null){
			list = features.asStringList();
			dbo.put("features", list);
		}
		dbo.put("classify", classify);
		dbo.put("baseDir", baseDir);
		dbo.put("tree", root.asMap());
		return dbo;
	}
	
	protected SplitNode<E> createSplitNode() {
		return new SplitNode<E>(this, 1, this.trainingDataSet);
	}

	public static Map<String, List<Double>> calculateSplitsPerFeature(
			Features features, DataSet<? extends Instance> data,
			double tolerance) {
		Map<String, List<Double>> splitsPerFeature  = new HashMap<String, List<Double>>();
		for(int f = 0; f < features.size(); f++){
			String feature = features.getFeatureByIndex(f);
			List<Double> distincts = data.distinct(feature, tolerance);
			List<Double> splits = new ArrayList<Double>();
			Collections.sort(distincts);
			for(int d = 0; d < distincts.size() - 1; d++)
				splits.add(distincts.get(d)
						+ (distincts.get(d + 1) - distincts.get(d)) / 2.0);
			Log.puts(feature + " splits: " + splits);
			splitsPerFeature.put(feature, splits);
		}
		return splitsPerFeature;
	}

	@Override
	public DataSet<E> test() {
		for(Instance each : testDataSet){
			root.test(each);
		}
		return this.testDataSet;
	}

	public synchronized Random getRandom() {
		return new Random(rand.nextInt(1000));
	}

	public List<Double> getSplitsForFeature(String feature) {
		return this.splitsPerFeature.get(feature);
	}

}
