package ch.eonum.pipeline.transformation;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.SparseInstance;
import ch.eonum.pipeline.util.Log;

/**
 * Nearest Neighbor imputation: Replace all missing values of an instance,
 * represented by Double.NaN, by the corresponding value of its nearest
 * neighbor. Instances with NaN values are not taken as nearest neighbors.
 * 
 * @author tim
 * 
 */
public class NearestNeighborImputation<E extends Instance> extends Transformer<E> {

	private DataSet<SparseInstance> neighbors;
	private Features features;

	public NearestNeighborImputation(DataSet<SparseInstance> dataTraining, Features features) {
		this.neighbors = dataTraining;
		this.features = features;
	}
	
	@Override
	public void extract(){
		super.extract();
		int t = 0;
		int j = 0;
		for(Instance each : this.dataSet){
			t++;
			Instance neighbor = null;
			for(int i = 0; i < features.size(); i++){
				String feature = features.getFeatureByIndex(i);
				if(Double.isNaN(each.get(feature))){
					if(neighbor == null){
						neighbor = getNeighbor(each);
						j++;
						Log.puts(t + " / " + j);
					}
					each.put(feature, neighbor.get(feature));
				}
			}
		}
	}
	

	private Instance getNeighbor(Instance inst) {
		Instance nearestNeighbor = null;
		double minDistance = Double.POSITIVE_INFINITY;
		for(Instance each : neighbors){
			if(each == inst) continue;
			double dist = distance(each, inst);	
			if(dist < minDistance){
				minDistance = dist;
				nearestNeighbor = each;
			}
		}
		return nearestNeighbor;
	}

	/**
	 * Euclidian distance handling NaN values
	 * NaN values in the second instance are being ignored.
	 * NaN values in the first instance lead to a infinite distance.
	 * 
	 * @param each
	 * @param inst
	 * @return
	 */
	private double distance(Instance inst1, Instance inst2) {
		double distance = 0.0;
		for (int i = 0; i < features.size(); i++) {
			String feature = features.getFeatureByIndex(i);
			if (Double.isNaN(inst1.get(feature)))
				return Double.POSITIVE_INFINITY;
			if (Double.isNaN(inst2.get(feature)))
				continue;
			distance += Math.pow(
					Math.abs(inst2.get(feature) - inst1.get(feature)), 2.);
		}

		return distance;
	}

}
