package ch.eonum.pipeline.classification.meta;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.util.FileUtil;
import ch.eonum.pipeline.util.Log;

/**
 * AdaBoost.M1 for binary classification.
 * 
 * @author tim
 *
 */
public class AdaBoost<E extends Instance> extends Classifier<E> {
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {
		PARAMETERS.put("m", "number of iterations (default 10)");
	}

	protected Classifier<E> baseClassifier;
	protected double[] modelWeights;
	protected PrintStream printer;
	public AdaBoost(Classifier<E> baseClassifier, Features features) {
		this.features = features;
		this.setSupportedParameters(AdaBoost.PARAMETERS);
		this.putParameter("m", 10.0);
		this.baseClassifier = baseClassifier;
		
	}

	@Override
	public void train() {
		prepareClasses();
		try {
			this.printer = new PrintStream(new File(baseDir + "adaBoost.log"));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		for (Instance each : trainingDataSet)
			each.weight = 1.0;
		for (Instance each : testDataSet)
			each.weight = 1.0;
		
		int M = (int) this.getDoubleParameter("m");
		this.modelWeights = new double[M];
		
		double maxOutcome = 0.0;
		for(Instance each : trainingDataSet)
			maxOutcome = Math.max(maxOutcome, each.outcome);
		
		for(int iteration = 0; iteration < M; iteration++){			
			/** fit the model. */
			baseClassifier.setTrainingSet(trainingDataSet);
			baseClassifier.setTestSet(testDataSet);
			FileUtil.mkdir(baseDir + iteration + "/");
			baseClassifier.setBaseDir(baseDir + iteration + "/");
			baseClassifier.train();
			
			/** test. */
			baseClassifier.setTestSet(trainingDataSet);
			baseClassifier.test();
			baseClassifier.setTestSet(testDataSet);
			baseClassifier.test();
			
			/** calculate error. */
			double error = 0.0;
			double weightSum = 0.0;
			for(Instance each : trainingDataSet){
				weightSum += each.weight;
				error += each.weight * getWeight(maxOutcome, each);
			}
			error /= weightSum;
			modelWeights[iteration] = Math.log((1 - error)/error);
			
			/** set new weights. */
			for (Instance each : trainingDataSet)
				each.weight = each.weight
						* Math.exp(modelWeights[iteration]
								* getWeight(maxOutcome, each));
			for (Instance each : testDataSet)
				each.weight = each.weight
						* Math.exp(modelWeights[iteration]
								* getWeight(maxOutcome, each));

			
			
			Log.puts("Iteration " + iteration + ": Error: " + error + " Model weight: " + modelWeights[iteration]);
			printer.println("Iteration " + iteration + ": Error: " + error + " Model weight: " + modelWeights[iteration]);
		}
	}

	protected double getWeight(double maxOutcome, Instance each) {
		return (each.label.equals(each.groundTruth)) ? 0.0 : 1.0;
	}
	
	@Override
	public DataSet<E> test() {
		for(E e : testDataSet){
			e.putResult("result", 0.);
			for(String className : classes.asSet())
				e.putResult("result" + className + "Temp", 0.);
		}
		double weightSum = 0.0;
		int M = (int) this.getDoubleParameter("m");
		for(int iteration = 0; iteration < M; iteration++){
			weightSum += modelWeights[iteration];
			baseClassifier.setTestSet(testDataSet);
			baseClassifier.setBaseDir(baseDir + iteration + "/");
			testDataSet = baseClassifier.test();
			for(Instance each : testDataSet)
				for(String className : classes.asSet())
					each.putResult("result" + className + "Temp",
							each.getResult("result" + className + "Temp")
									+ each.getResult("result" + className)
									* modelWeights[iteration]);
				
		}
		for(Instance each : testDataSet){
			double max = Double.NEGATIVE_INFINITY;
			String maxClass = null;
			for(String className : classes.asSet()){
				double result = each.getResult("result" + className + "Temp") / weightSum;
				each.putResult("result" + className, result);
				each.putResult("result" + className + "Temp", 0.);
				if(result > max){
					max = result;
					maxClass = className;
				}
			}
			each.label = maxClass;
			each.put("result", max);
		}
		
		return this.testDataSet;
	}
}
