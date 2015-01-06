package ch.eonum.pipeline.features;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.Parameters;
import ch.eonum.pipeline.evaluation.Evaluator;
import ch.eonum.pipeline.util.Gnuplot;
import ch.eonum.pipeline.util.Log;

/**
 * Feature importance ranking. For each feature, all values are randomly
 * permuted among all test instances. The decrease or increase in accuracy after
 * this operation is assigned as the feature importance for this feature.
 * 
 * @author tim
 * 
 */
public class FeatureImportance<E extends Instance> extends Parameters {
	private static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {}

	private Classifier<E> classifier;
	private Features features;
	private Evaluator<E> evaluator;
	private List<FeatureDelta> featureDeltas;
	private DataSet<E> testData;
	
	/**
	 * Constructor
	 * @param classifier underlying classifier
	 * @param test test set
	 * @param eval evaluator
	 * @param features feature set
	 */
	public FeatureImportance(Classifier<E> classifier, DataSet<E> test, Evaluator<E> eval, Features features) {
		this.classifier = classifier;
		this.features = features;
		this.evaluator = eval;
		this.testData = test;
		this.setSupportedParameters(FeatureImportance.PARAMETERS);	
	}

	/**
	 * Feature importance ranking. A file called "positiveFeatures.txt" with the
	 * analysis results is written to the base directory of the provided
	 * classifier.
	 */
	public void createRanking() {
		Random rand = new Random(123);
		featureDeltas = new ArrayList<FeatureDelta>();
		double baseLine = evaluate(testData);
		for(int f = 0; f < features.size(); f++){
			String feature = features.getFeatureByIndex(f);
			DataSet<E> copy = testData.deepCopy();
			for (Instance each : copy)
				each.exchangeFeature(feature, copy.get(rand.nextInt(copy.size())));
			double eval = evaluate(copy);
			featureDeltas.add(new FeatureDelta(feature, baseLine - eval));
			Log.puts("Ranking feature nr. " + f + ": " + feature + " Delta: " + (baseLine - eval));
		}
		Collections.sort(featureDeltas);
		printRanking();
		
		validateNumberOfFeatures();
	}

	/**
	 * Additional analysis. Incrementally use only the best features for
	 * classification. All other features are randomly permutated. A file called
	 * "validationOnTheNumberOfFeatures.png" is written to the classifier's base
	 * directory.
	 */
	private void validateNumberOfFeatures() {
		Random rand = new Random(123);
		Map<Integer, Double> curve = new LinkedHashMap<Integer, Double>();
		List<String> bestFeatures = new ArrayList<String>();
		int i = 0;
		for(FeatureDelta each : featureDeltas){
			bestFeatures.add(each.getFeature());
			DataSet<E> copy = testData.deepCopy();
			for(int f = 0; f < features.size(); f++){
				String feature = features.getFeatureByIndex(f);
				if(!bestFeatures.contains(feature))
					for (Instance inst : copy)
						inst.exchangeFeature(feature, copy.get(rand.nextInt(copy.size())));
			}
			double eval = evaluate(copy);
			curve.put(++i, eval);
			Log.puts("Validating feature nr. " + i + " Eval: " + eval);
			Gnuplot.plotOneDimensionalCurve(curve, "Validation N",
					classifier.getBaseDir()
							+ "validationOnTheNumberOfFeatures.png");
		}
	}

	private void printRanking() {
		int rank = 0;
		Features positive = new Features();
		for(FeatureDelta each : featureDeltas){
			Log.puts(++rank + ": " + each);
			if(each.getDelta() > 0)
				positive.addFeature(each.getFeature());
		}
		positive.recalculateIndex();
		positive.writeToFile(classifier.getBaseDir() + "positiveFeatures.txt");
	}

	private double evaluate(DataSet<E> data) {
		classifier.setTestSet(data);
		data = classifier.test();
		return evaluator.evaluate(data);
	}

}
