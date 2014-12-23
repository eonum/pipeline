package ch.eonum.pipeline.classification.tree;

import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.evaluation.Evaluator;
import ch.eonum.pipeline.util.FileUtil;
import ch.eonum.pipeline.util.Gnuplot;
import ch.eonum.pipeline.util.Log;

/**
 * Gradient boosting machine for decision trees (regression). Similar to @see
 * GradientBoosting but specialized on decision trees and hence more efficient.
 * 
 * @author tim
 * 
 */
public class GBM<E extends Instance> extends Classifier<E> {

	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();

	static {
		PARAMETERS.put("shrinkage", "shrink factor (default 1.0)");
		PARAMETERS.put("m", "number of trees (default 10.0)");
		PARAMETERS.put("k", "percentage of features which are taken into account at each split node [0,1] (default 0.333)");
		PARAMETERS.put("f", "fraction of the bootstrap size, relative to the training set size (default 1.0)");
		PARAMETERS.put("maxDepth", "maximum depth of the tree (default 10.0)");
		PARAMETERS.put("minSize", "minimum number of training instances within one terminal node (default 5.0)");
		PARAMETERS.put("tolerance", "tolerance range for distinct values (default 0.01)");
	}

	/** decision trees. */
	protected List<DecisionTree<E>> trees;
	private Evaluator<E> evaluator;
	private double[] gammas;
	private Random rand;

	public GBM(Features features, Evaluator<E> evaluator, int seed) {
		this.evaluator = evaluator;
		this.setSupportedParameters(GBM.PARAMETERS);
		this.putParameter("shrinkage", 1.0);
		this.putParameter("m", 10.0);
		this.putParameter("k", 0.333);
		this.putParameter("maxDepth", 10.0);
		this.putParameter("minSize", 15.0);
		this.putParameter("tolerance", 0.01);
		this.putParameter("f", 1.0);
		this.rand = new Random(seed);
		this.features = features;
	}

	@Override
	public void train() {
		Map<Instance, Double> originalOutcomes = saveOutcomes(this.trainingDataSet);
		Map<Integer, Double> curve = new LinkedHashMap<Integer, Double>();
		
		
		Map<String, List<Double>> splitsPerFeature = DecisionTree.calculateSplitsPerFeature(features,
				trainingDataSet, this.getDoubleParameter("tolerance"));
		
		int M = (int)this.getDoubleParameter("m");
		double shrinkage = this.getDoubleParameter("shrinkage");
		trees = new ArrayList<DecisionTree<E>>();
		for(int i = 0; i < M-1; i++){
			DecisionTree<E> dt = new DecisionTree<E>(features, rand.nextInt(), splitsPerFeature, i);
			dt.putParameter("k", this.getDoubleParameter("k"));
			dt.putParameter("maxDepth", this.getDoubleParameter("maxDepth"));
			dt.putParameter("minSize", this.getDoubleParameter("minSize"));
			trees.add(dt);
		}
		
		// F_0 average outcome
		double avg = 0.0;
		for (Instance each : trainingDataSet)
			avg += each.outcome;
		avg /= trainingDataSet.size();
		for (Instance each : trainingDataSet)
			each.putResult("result", avg);
				
		this.gammas = new double[M];
		gammas[0] = avg;
		// F_m minus 1
		Map<Instance, Double> F_m1 = saveResults(this.trainingDataSet);
		
		PrintStream p;
		try {
			p = new PrintStream(this.getBaseDir() + "gradientBoosting.txt");
			p.println("Gamma[0] = " + avg);

			for (int m = 1; m < M; m++) {
				DecisionTree<E> tree = trees.get(m-1);
				
				DataSet<E> trainSet = bootstrap(trainingDataSet);
				DataSet<E> testSet = outOfBag(trainSet, trainingDataSet);
				// calculate residuals and set outcome/target to residual
				for (Instance each : trainingDataSet)
					each.outcome = originalOutcomes.get(each) - F_m1.get(each);
				
				FileUtil.mkdir(baseDir + m + "/");
				tree.setBaseDir(baseDir + m + "/");
				tree.setTrainingSet(trainSet);
				tree.train();
				tree.setTestSet(trainingDataSet);
				trainingDataSet = tree.test();
				trainingDataSet.renameResults("result", "Fm");
				
				loadOutComes(this.trainingDataSet, originalOutcomes);

				double maxGamma = 0.0;
				double maxEval = Double.NEGATIVE_INFINITY;
				for (int i = 0; i < 150; i++) {
					double gamma = i * 0.01;
					for (Instance each : testSet)
						each.putResult("result",
								each.getResult("Fm") * gamma + F_m1.get(each));
					double eval = this.evaluator.evaluate(testSet);
//					Log.puts("Gamma: " + gamma + " Eval: " + eval);
					if (eval > maxEval) {
						maxEval = eval;
						maxGamma = gamma;
					}
				}
				Log.puts("Iteration " + m + " Maximum Gamma: " + maxGamma
						+ " Eval: " + maxEval);

				gammas[m] = maxGamma * shrinkage;
				for (Instance each : trainingDataSet)
					each.putResult("result", (each.getResult("Fm"))
							* gammas[m] + F_m1.get(each));
				
				F_m1 = saveResults(trainingDataSet);

				double eval = evaluator.evaluate(testSet);

				p.println("Iteration " + m + " Maximum Gamma: " + maxGamma
						+ " Eval: " + maxEval + " max eval shrinked: " + eval);
				curve.put(m, eval);
				Gnuplot.plotOneDimensionalCurve(curve, "GBM", this.getBaseDir() + "gbm");
			}
			
			p.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		loadOutComes(this.trainingDataSet, originalOutcomes);
	}
	
	private <T extends Instance> DataSet<T> bootstrap(DataSet<T> data) {
		DataSet<T> bootstrap = new DataSet<T>();
		int size = (int) (this.getDoubleParameter("f") * data.size());
		for(int i = 0; i < size; i++)
			bootstrap.addInstance(data.get(rand.nextInt(data.size())));
		return bootstrap;
	}
	
	/**
	 * OOB, out of bag
	 * @param bag
	 * @param all
	 * @return
	 */
	private <T extends Instance> DataSet<T> outOfBag(DataSet<T> bag, DataSet<T> all) {
		DataSet<T> oob = new DataSet<T>();
		for(T each : all)
			if(!bag.contains(each))
				oob.addInstance(each);
		return oob;
	}

	private Map<Instance, Double> saveResults(DataSet<E> data) {
		Map<Instance, Double> map = new HashMap<Instance, Double>();
		for (Instance each : data)
			map.put(each, (double)each.getResult("result"));
		return map;
	}

	private void loadOutComes(DataSet<E> data,
			Map<Instance, Double> originalOutcomes) {
		for (Instance each : data)
			each.outcome = (double)originalOutcomes.get(each);
	}

	private Map<Instance, Double> saveOutcomes(DataSet<E> data) {
		Map<Instance, Double> map = new HashMap<Instance, Double>();
		for (Instance each : data)
			map.put(each, each.outcome);
		return map;
	}

	@Override
	public DataSet<E> test() {
		int M = (int) this.getDoubleParameter("m");
		for (Instance each : testDataSet)
			each.putResult("gbmSum", gammas[0]);
		
		for (int m = 1; m < M; m++) {
			DecisionTree<E> tree = trees.get(m-1);
			tree.setTestSet(testDataSet);
			tree.setBaseDir(baseDir + m + "/");
			this.testDataSet = tree.test();
			for(Instance each : testDataSet)
				each.putResult("gbmSum", each.getResult("gbmSum") + gammas[m] * (each.getResult("result")));
		}
		for (Instance each : testDataSet){
			each.putResult("result", Math.max(0.0, each.getResult("gbmSum")));
			each.removeResult("gbmSum");
		}
		return this.testDataSet;
	}
}
