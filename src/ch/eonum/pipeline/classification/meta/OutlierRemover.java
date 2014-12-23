package ch.eonum.pipeline.classification.meta;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;

/**
 * Remove outliers in the training set and retrain the underlying model.
 * (Instances with a high error)
 * 
 * @author tim
 * 
 * @param <E>
 */
public class OutlierRemover<E extends Instance> extends Classifier<E> {

	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {
		PARAMETERS.put("percentage", "percentage of outliers which are to be removed from the training data set. (default: 1.0%)");
	}

	private Classifier<E> baseClassifier;

	public OutlierRemover(Classifier<E> baseClassifier) {
		this.baseClassifier = baseClassifier;
		this.testDataSet = this.baseClassifier.getTestDataSet();
		this.trainingDataSet = this.baseClassifier.getTrainingDataSet();
		this.setSupportedParameters(OutlierRemover.PARAMETERS);
		this.putParameter("percentage", 1.0);
	}

	@Override
	public void train() {
		this.baseClassifier.setTrainingSet(trainingDataSet);
		this.baseClassifier.setTestSet(testDataSet);
		this.baseClassifier.train();
		this.baseClassifier.setTestSet(trainingDataSet);
		this.baseClassifier.test();
		this.baseClassifier.setTestSet(testDataSet);
		DataSet<E> reduced = new DataSet<E>();
		for(E each : this.trainingDataSet){
			each.putResult("error", Math.abs(each.getResult("result")-each.outcome));
			reduced.add(each);
		}
		Collections.sort(reduced, new Comparator<Instance>(){
			@Override
			public int compare(Instance arg0, Instance arg1) {
				return (arg0.getResult("error") - arg1.getResult("error")) > 0.0 ? 1 : -1;
			}		
		});
		
		/** remove the instances with the largest error. */
		double p = this.getDoubleParameter("percentage");
		int numOutliers = (int)((p/100.0) * reduced.size());
		for(int i = numOutliers; i > 0; i--)
			reduced.removeElementAt(reduced.size() - 1);
		
		this.baseClassifier.setTrainingSet(reduced);
		for(E e : trainingDataSet)
			e.putResult("error", 0.);
		this.baseClassifier.train();
		this.baseClassifier.setTrainingSet(trainingDataSet);
	}
	
	@Override
	public DataSet<E> test() {
		baseClassifier.setTestSet(testDataSet);
		return baseClassifier.test();
	}
	
	@Override
	public void setFeatures(Features features){
		super.setFeatures(features);
		this.baseClassifier.setFeatures(features);
	}

}
