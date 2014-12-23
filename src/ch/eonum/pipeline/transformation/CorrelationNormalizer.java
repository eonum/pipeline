package ch.eonum.pipeline.transformation;

import java.util.HashMap;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.SparseInstance;
import ch.eonum.pipeline.util.Log;

/**
 * Multiply each feature with it's correlation to the outcome.
 * 
 * @author tim
 *
 */
public class CorrelationNormalizer<E extends Instance> extends Transformer<E> {
	private SparseInstance correlation;

	/**
	 * create a normalizer from a given dataset.
	 * correlations are calculated out of the dataset.
	 * @param dataset
	 */
	public CorrelationNormalizer(DataSet<E> dataset) {
		this.prepare(dataset);
	}
	
	@Override
	public void prepare(DataSet<E> dataset){
		this.correlation = new SparseInstance("", "", new HashMap<String, Double>());
		double avgGt = 0.0;
		for(Instance each : dataset)
			avgGt += each.outcome;
		avgGt /= dataset.size();
		
		double sumSquareDeltasGt = 0.0;
		for(Instance each : dataset)
			sumSquareDeltasGt += Math.pow(each.outcome - avgGt, 2);

		for(String feature : dataset.features()){
			double avg = 0.0;
			for(Instance each : dataset)
				avg += each.get(feature);
			avg /= dataset.size();
			
			double covariance = 0.0;
			double variance = 0.0;
			for(Instance each : dataset){
				double fDelta = each.get(feature) - avg;
				double gtDelta = each.outcome - avgGt;
				covariance += fDelta * gtDelta;
				variance += Math.pow(fDelta, 2);
			}
			
			double corr = covariance / (Math.sqrt(variance*sumSquareDeltasGt));
			if(Double.isNaN(corr)) corr = 0.0;
			this.correlation.put(feature, Math.abs(corr));
			Log.puts("Processed feature: " + feature + " Correlation: " + corr);
		}
	}

	@Override
	public void extract(){
		super.extract();
		for(Instance each : this.dataSet)
			each.times(this.correlation);
	}

}
