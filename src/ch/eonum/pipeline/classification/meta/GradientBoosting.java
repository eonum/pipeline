package ch.eonum.pipeline.classification.meta;

import java.io.IOException;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.evaluation.Evaluator;
import ch.eonum.pipeline.util.FileUtil;
import ch.eonum.pipeline.util.Gnuplot;
import ch.eonum.pipeline.util.Log;

/**
 * Gradient Boosting Machine for regression tasks.
 * 
 * @author tim
 * 
 */
public class GradientBoosting<E extends Instance> extends Classifier<E> {
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();

	static {
		PARAMETERS.put("shrinkage", "shrink factor (default 1.0)");
	}

	protected List<Classifier<E>> baseClassifiers;
	private Evaluator<E> evaluator;
	private double[] gammas;
	private double[] deltas;
	private double[] ranges;
	private Random rand;

	public GradientBoosting(List<Classifier<E>> baseClassifiers, Evaluator<E> evaluator, int seed) {
		this.baseClassifiers = baseClassifiers;
		this.evaluator = evaluator;
		this.setSupportedParameters(GradientBoosting.PARAMETERS);
		this.putParameter("shrinkage", 1.0);
		this.rand = new Random(seed);
	}

	@Override
	public void train() {
		Map<Instance, Double> originalOutcomes = saveOutcomes(this.trainingDataSet);
		Map<Integer, Double> curve = new LinkedHashMap<Integer, Double>();
		
		/** F_0 average outcome. */
		double avg = 0.0;
		for(Instance each : trainingDataSet)
			avg += each.outcome;
		avg /= trainingDataSet.size();
		for(Instance each : trainingDataSet)
			each.putResult("result", avg);

		int M = baseClassifiers.size() + 1;
		double shrinkage = this.getDoubleParameter("shrinkage");
		
		this.gammas = new double[M];
		this.deltas = new double[M];
		this.ranges = new double[M];
		gammas[0] = avg;
		deltas[0] = 0.0;
		ranges[0] = 1.0;
		// F_m minus 1
		Map<Instance, Double> F_m1 = saveResults(this.trainingDataSet);
		
		PrintStream p;
		try {
			p = new PrintStream(this.getBaseDir() + "gradientBoosting.txt");
		p.println("Gamma[0] = " + avg);
		
		for(int m = 1; m < M; m++){
			Classifier<E> c = this.baseClassifiers.get(m-1);
			DataSet<E> trainSet = bootstrap(trainingDataSet);
			DataSet<E> testSet = outOfBag(trainSet, trainingDataSet);
			/** calculate residuals and set outcome/target to residual. */
			double deltaMin = Double.POSITIVE_INFINITY;
			double deltaMax = Double.NEGATIVE_INFINITY;
			for(Instance each : trainingDataSet){
				each.outcome = originalOutcomes.get(each) - F_m1.get(each);
				deltaMin = Math.min(each.outcome, deltaMin);
				deltaMax = Math.max(each.outcome, deltaMax);
			}
			ranges[m] = deltaMax -deltaMin;
			for(Instance each : trainingDataSet){
				each.outcome -= deltaMin;
				each.outcome /= ranges[m];
			}
			
			deltas[m] = deltaMin;
			FileUtil.mkdir(baseDir + m + "/");
			c.setTrainingSet(trainSet);
			c.setTestSet(testSet);
			FileUtil.mkdir(c.getBaseDir());
			c.setFeatures(getFeatures());
			c.train();
			c.setTestSet(trainingDataSet);
			trainingDataSet = c.test();	
			trainingDataSet.renameResults("result", "Fm");
			
			loadOutComes(this.trainingDataSet, originalOutcomes);
			
			double maxGamma = 0.0;
			double maxEval = Double.NEGATIVE_INFINITY;
			for(int i = 0; i < 150; i ++){
				double gamma = i * 0.01;
				for(Instance each : testSet)
					each.putResult("result", (each.getResult("Fm") * ranges[m] + deltaMin) * gamma + F_m1.get(each));
				double eval = this.evaluator.evaluate(testSet);
				if(eval > maxEval){
					maxEval = eval;
					maxGamma = gamma;
				}
			}
			Log.puts("Iteration " + m + " Maximum Gamma: " + maxGamma + " Eval: " + maxEval);
			
			gammas[m] = maxGamma * shrinkage;
			for(Instance each : trainingDataSet)
				each.putResult("result", (each.getResult("Fm") * ranges[m] + deltaMin) * gammas[m] + F_m1.get(each));
			F_m1 = saveResults(trainingDataSet);
			
			double eval = evaluator.evaluate(testSet);
			
			p.println("Iteration " + m + " Maximum Gamma: " + maxGamma + " Eval: " + maxEval + " max eval shrinked: " + eval);
			p.println("Delta for iteration " + m + ": " + deltaMin + " Range: " + ranges[m]);
			curve.put(m, eval);
			Gnuplot.plotOneDimensionalCurve(curve, "GBM", this.getBaseDir() + "gbm");
		}
		loadOutComes(this.trainingDataSet, originalOutcomes);
		p.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private DataSet<E> bootstrap(DataSet<E> data) {
		DataSet<E> bootstrap = new DataSet<E>();
		for(int i = 0; i < data.size(); i++)
			bootstrap.addInstance(data.get(rand.nextInt(data.size())));
		return bootstrap;
	}
	
	/**
	 * OOB, out of bag
	 * @param bag
	 * @param all
	 * @return
	 */
	private DataSet<E> outOfBag(DataSet<E> bag, DataSet<E> all) {
		DataSet<E> oob = new DataSet<E>();
		for(E each : all)
			if(!bag.contains(each))
				oob.addInstance(each);
		return oob;
	}

	private Map<Instance, Double> saveResults(DataSet<E> data) {
		Map<Instance, Double> map = new HashMap<Instance, Double>();
		for (Instance each : data)
			map.put(each, each.getResult("result"));
		return map;
	}

	private void loadOutComes(DataSet<E> data,
			Map<Instance, Double> originalOutcomes) {
		for (Instance each : data)
			each.outcome = originalOutcomes.get(each);
	}

	private Map<Instance, Double> saveOutcomes(DataSet<E> data) {
		Map<Instance, Double> map = new HashMap<Instance, Double>();
		for (Instance each : data)
			map.put(each, each.outcome);
		return map;
	}

	@Override
	public DataSet<E> test() {
		int M = baseClassifiers.size() + 1;
		for (Instance each : testDataSet)
			each.putResult("gbmSum", gammas[0]);
		
		for (int m = 1; m < M; m++) {
			Classifier<E> c = baseClassifiers.get(m-1);
			c.setTestSet(testDataSet);
			this.testDataSet = c.test();
			for(Instance each : testDataSet)
				each.putResult("gbmSum", each.getResult("gbmSum") + gammas[m]
						* (each.getResult("result") * ranges[m] + deltas[m]));
		}
		for (Instance each : testDataSet){
			each.putResult("result", Math.max(0.0, each.getResult("gbmSum")));
			each.putResult("gbmSum", 0.);
		}
		return this.testDataSet;
	}

}
