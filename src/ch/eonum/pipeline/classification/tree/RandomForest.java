package ch.eonum.pipeline.classification.tree;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.classification.meta.Bagging;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;

/**
 * Random Forest. @link http://en.wikipedia.org/wiki/Random_forest An ensemble
 * of decision trees for regression. Extends the standard Bagging ensemble
 * method by decision tree specific parameters.
 * 
 * @author tim
 * 
 * @param <E>
 */
public class RandomForest<E extends Instance> extends Bagging<E> {
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {
		PARAMETERS.put("k", "percentage of features which are taken into account at each split node [0,1] (default 0.333)");
		PARAMETERS.put("maxDepth", "maximum depth of the tree (default 10.0)");
		PARAMETERS.put("minSize", "minimum number of training instances within one terminal node (default 5.0)");
		PARAMETERS.put("tolerance", "tolerance range for distinct values (default 0.01)");
		PARAMETERS.put("numTrees", "number of trees in the forest (default 200.0)");
	}
	
	protected Map<String, List<Double>> splitsPerFeature;

	public RandomForest(Features features, int seed) {
		super(new ArrayList<Classifier<E>>(), features, seed);
		this.setSupportedParameters(RandomForest.PARAMETERS);
		this.putParameter("k", 0.333);
		this.putParameter("maxDepth", 10.0);
		this.putParameter("minSize", 15.0);
		this.putParameter("tolerance", 0.01);
		this.putParameter("numTrees", 100.0);
	}
	
	/**
	 * parallel training of all decision trees.
	 */
	@Override
	public void train() {
		splitsPerFeature = DecisionTree.calculateSplitsPerFeature(features,
				trainingDataSet, this.getDoubleParameter("tolerance"));
		for(int i = 0; i < this.getDoubleParameter("numTrees"); i++){
			DecisionTree<E> dt = createDecisionTree(i);
			dt.putParameter("k", this.getDoubleParameter("k"));
			dt.putParameter("maxDepth", this.getDoubleParameter("maxDepth"));
			dt.putParameter("minSize", this.getDoubleParameter("minSize"));
			DataSet<E> train = bootstrap(trainingDataSet);
			dt.setTrainingSet(train);
			dt.setTestSet(outOfBag(train, trainingDataSet));
			baseClassifiers.add(dt);
		}
		
		ExecutorService service = Executors.newFixedThreadPool(Math.min(Runtime
				.getRuntime().availableProcessors(), baseClassifiers.size()));
		
		for(Classifier<E> dt : baseClassifiers)
			service.submit(dt);
		
		service.shutdown();
		try {
			service.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);
		} catch (InterruptedException e) {
			e.printStackTrace();
			System.exit(-1);
		}

	}


	protected DecisionTree<E> createDecisionTree(int i) {
		return new DecisionTree<E>(features, rand.nextInt(), splitsPerFeature, i);
	}
}
