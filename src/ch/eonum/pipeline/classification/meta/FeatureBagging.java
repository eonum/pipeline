package ch.eonum.pipeline.classification.meta;

import java.util.HashMap;
import java.util.Map;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.util.FileUtil;

/**
 * Bagging ensemble.
 * 
 * @author tim
 *
 */
public class FeatureBagging<E extends Instance> extends Classifier<E> {
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {
		PARAMETERS.put("k", "number of feature subset or base classifiers, size of the ensemble (default 5.0)");
		PARAMETERS.put("p", "probability of a feature to be in the base classifier (p/k) (default 1.0)");
	}

	private Classifier<E> baseClassifier;
	private Features[] featureSubsets;

	public FeatureBagging(Classifier<E> baseClassifier, Features features) {
		this.setFeatures(features);
		this.baseClassifier = baseClassifier;
		this.setSupportedParameters(FeatureBagging.PARAMETERS);
		this.putParameter("k", 5.0);
		this.putParameter("p", 1.0);
	}

	@Override
	public void train() {
		this.baseClassifier.setTrainingSet(trainingDataSet);
		this.baseClassifier.setTestSet(testDataSet);
		int k = (int)this.getDoubleParameter("k");
		this.featureSubsets = new Features[k];
		for(int i = 0; i < k; i++){
			featureSubsets[i] = this.getFeaturesSubset(i,k);
			FileUtil.mkdir(baseDir + "/" + i);
			baseClassifier.setFeatures(featureSubsets[i]);
			featureSubsets[i].writeToFile(baseDir + "/" + i + "/features.txt");
			baseClassifier.setBaseDir(baseDir + "/" + i + "/");
			baseClassifier.train();
		}
	}
	
	private Features getFeaturesSubset(int i, int k) {
		Features f = new Features();
		
		double p = this.getDoubleParameter("p")/k;
		for (String feature : this.features.getListOfFeaturesCopy()){
			double rand = Math.random();
			if (rand < p)
				f.addFeature(feature);
		}
		f.recalculateIndex();
		return f;
	}

	@Override
	public DataSet<E> test(){
		int k = (int)this.getDoubleParameter("k");
		for(E e : testDataSet){
			e.putResult("result", 0.);
			e.putResult("resultTemp", 0.);
		}
		for(int i = 0; i < k; i++){
			baseClassifier.setBaseDir(baseDir + "/" + i + "/");
			baseClassifier.setTestSet(testDataSet);
			baseClassifier.setFeatures(this.featureSubsets[i]);
			testDataSet = baseClassifier.test();
			for(Instance each : testDataSet)
				each.putResult("resultTemp", each.getResult("resultTemp") + each.getResult("result"));
		}
		for(Instance each : testDataSet)
			each.putResult("result", each.getResult("resultTemp") / (double) k);
		return this.testDataSet;
	}

}
