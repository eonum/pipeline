package ch.eonum.pipeline.classification;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;

/**
 * A simple regressor for benchmarking purposes:
 * Set the probability measure to be one certain feature.
 * 
 * @author tim
 *
 */
public class SingleFeatureClassifier<E extends Instance> extends Classifier<E> {

	private String feature;
	
	public SingleFeatureClassifier(String feature){
		this.feature = feature;
	}

	@Override
	public DataSet<E> test() {
		for (Instance inst : this.testDataSet)
			inst.putResult("result", inst.get(this.feature));
		return this.testDataSet;
	}

	@Override
	public void train() {}

}
