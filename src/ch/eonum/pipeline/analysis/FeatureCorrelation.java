package ch.eonum.pipeline.analysis;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.util.Log;

/**
 * calculate the correlation between each feature.
 * 
 * @author tim
 *
 */
public class FeatureCorrelation<E extends Instance> {

	public void printCorrelations(DataSet<E> data, Features features) {
		for(int i = 0; i < features.size(); i++)
			for(int j = 0; j < features.size(); j++)
				if(i != j)
					this.correlate(data, features.getFeatureByIndex(i),
							features.getFeatureByIndex(j));
	}

	private void correlate(DataSet<E> data, String f1,
			String f2) {
		double avg1 = 0.0;
		double avg2 = 0.0;
		for(Instance each : data){
			avg1 += each.get(f1);
			avg2 += each.get(f2);
		}
		avg1 /= data.size();
		avg2 /= data.size();
		
		double covariance = 0.0;
		double variance1 = 0.0;
		double variance2 = 0.0;
		for(Instance each : data){
			double delta1 = each.get(f1) - avg1;
			double delta2 = each.get(f2) - avg2;
			covariance += delta1 * delta2;
			variance1 += Math.pow(delta1, 2);
			variance2 += Math.pow(delta2, 2);
		}
		
		double corr = covariance / (Math.sqrt(variance1 * variance2));
		Log.puts(f1 + " : " + f2 + " => " + corr);
	}

}
