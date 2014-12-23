package ch.eonum.pipeline.classification.meta;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;

/**
 * Median Bagging. Get the median regression result from a set of regressors.
 * 
 * @author tim
 *
 * @param <E>
 */
public class MedianBagging<E extends Instance> extends Bagging<E> {

	public MedianBagging(List<Classifier<E>> baseClassifiers, Features features, int seed) {
		super(baseClassifiers, features, seed);
	}
	
	@Override
	public DataSet<E> test(){
		for(E e : testDataSet){
			e.putResult("result", 0.);
			e.putResult("resultTemp", 0.);
		}
		int k = 0;
		for(Classifier<E> dt : baseClassifiers){
			dt.setTestSet(testDataSet);
			testDataSet = dt.test();
			for(Instance each : testDataSet)
				each.putResult("resultTemp" + k++, each.getResult("result"));
		}
		
		for(Instance each : testDataSet){
			List<Double> sortedResults = new ArrayList<Double>();
			for(int i = 0; i < k; i++){
				sortedResults.add(each.getResult("resultTemp" + i));
				each.putResult("resultTemp" + i, 0.);
			}
			Collections.sort(sortedResults);
			int medianIndex = sortedResults.size()/2;
			each.putResult("result", sortedResults.get(medianIndex));
		}
		return this.testDataSet;
	}

}
