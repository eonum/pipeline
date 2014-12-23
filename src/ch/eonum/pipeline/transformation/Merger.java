package ch.eonum.pipeline.transformation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;


/**
 * merges two datasets according their features.
 * if some features are in more than one dataset, the last set is used.
 * @author tim
 *
 */
public class Merger<E extends Instance> extends Transformer<E> {
	private List<DataSet<E>> dataSets;
	private DataSet<E> result;
	
	public Merger(){
		this.dataSets = new ArrayList<DataSet<E>>();
	}

	@Override
	/**
	 * This method can be invoked more than once. 
	 */
	public void setInputDataSet(DataSet<E> dataSet) {
		this.addInputDataSet(dataSet);
	}

	@Override
	public DataSet<E> getOutputDataSet() {
		return result;
	}

	@Override
	public void extract() {
		Map<String, E> newData = new HashMap<String, E>();
		for (DataSet<E> each : dataSets) {
			for (E inst : each){
				if (newData.get(inst.id) == null){
					newData.put(inst.id, inst);
					for (String dimension : inst.features())
						addFeature(dimension);
				}
				else
					for (String dimension : inst.features()){
						addFeature(dimension);
						newData.get(inst.id).put(dimension,
								inst.get(dimension));
					}
			}
		}
		result = new DataSet<E>();
		result.addData(newData.values());
	}

	protected void resetDatasets() {
		this.dataSets = new ArrayList<DataSet<E>>();
	}

	
	@SafeVarargs
	public static <T extends Instance> Merger<T> merge(DataSet<T> ... dataSets) {
		Merger<T> merger = new Merger<T>();
		for(DataSet<T> each : dataSets)
			merger.addInputDataSet(each);
		merger.extract();
		return merger;
	}

	public void addInputDataSet(DataSet<E> dataSet) {
		this.dataSets.add(dataSet);
	}

}
