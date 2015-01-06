package ch.eonum.pipeline.transformation;

import java.util.HashMap;
import java.util.Map;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.Sequence;
import ch.eonum.pipeline.core.SparseInstance;

/**
 * Normalize a sequence data set to fit into the interval [0,1]. Only input
 * sequences are normalized.
 * 
 * Note: Dimensions/Features not present in the training set will not be
 * normalized at all.
 * 
 * 
 * @author tim
 * 
 */
public class MinMaxNormalizerSequence<E extends Sequence> extends Transformer<E> {
	private SparseInstance max;
	private SparseInstance min;
	private Features features;
	private Instance delta;

	/**
	 * create a normalizer from a given dataset.
	 * max and min are calculated out of the dataset.
	 * @param dataset
	 */
	public MinMaxNormalizerSequence(DataSet<E> dataset, Features features) {
		this.features = features;
		this.prepare(dataset);
	}
	
	@Override
	public void prepare(DataSet<E> dataset){
		max = new SparseInstance("", "", new HashMap<String, Double>());
		min = new SparseInstance("", "", new HashMap<String, Double>());
		for(Sequence s : dataset)
			for(int i = 0; i < features.size(); i++){
				String feature = features.getFeatureByIndex(i);
				for (int j = 0; j < s.getSequenceLength(); j++) {
					double value = s.get(j, feature); 
					if(max.get(feature) < value || !max.features().contains(feature))
						max.put(feature, value);
					if(min.get(feature) > value || !min.features().contains(feature))
						min.put(feature, value);
				}
			}
		min.cleanUp();
		max.cleanUp();
		
		delta = max.minusStateless(min);
		System.out.println("minimum instance" + min);
		System.out.println("maximum instance" + max);
		delta.cleanUp();
	}

	@Override
	public void extract(){
		super.extract();
		
		for(Sequence s : this.dataSet){
			for(int i = 0; i < features.size(); i++){
				String feature = features.getFeatureByIndex(i);
				for (int j = 0; j < s.getSequenceLength(); j++) {
					double value = (s.get(j, feature) - min.get(feature))
							/ delta.get(feature);
					s.put(j, feature, value);
				}
			}
			s.minus(min);
			s.divideBy(delta);
			s.cleanUp();
		}
	}

	public void normSingleTimePoint(Map<String, Double> point) {
		for (int i = 0; i < features.size(); i++) {
			String feature = features.getFeatureByIndex(i);
			double value = point.containsKey(feature) ? point.get(feature)
					: 0.0;
			value = (value - min.get(feature)) / delta.get(feature);
			point.put(feature, value);
		}
	}

}
