package ch.eonum.pipeline.transformation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.SparseInstance;

/**
 * combines the output of several classifiers.
 * All 'result' features are being extracted from the input data sets and combined in a new dataset.
 * You can use this for classifier combination.
 * 
 * @author tim
 *
 */
public class Combiner extends Transformer<SparseInstance> {
	private List<DataSet<SparseInstance>> dataSets;
	private DataSet<SparseInstance> result;
	
	public Combiner(){
		this.dataSets = new ArrayList<DataSet<SparseInstance>>();
	}

	@Override
	/**
	 * This method can be invoked more than once. 
	 */
	public void setInputDataSet(DataSet<SparseInstance> dataSet) {
		this.dataSets.add(dataSet);
	}

	@Override
	public DataSet<SparseInstance> getOutputDataSet() {
		return result;
	}

	@Override
	public void extract() {
		Map<String, SparseInstance> newData = new HashMap<String, SparseInstance>();
		int i = 0;
		for (DataSet<SparseInstance> each : dataSets) {
			i++;
			for (Instance inst : each){
				if(newData.containsKey(inst.id))
					newData.get(inst.id).putResult("result" + i, inst.getResult("result"));
				else {
					HashMap<String, Double> data = new HashMap<String, Double>();
					data.put("result" + i, inst.getResult("result"));
					SparseInstance newInst = new SparseInstance(inst.id, inst.groundTruth, data);
					newData.put(inst.id, newInst);
				}
			}
		}
		result = new DataSet<SparseInstance>();
		result.addData(newData.values());
	}
	
	protected void resetDatasets() {
		this.dataSets = new ArrayList<DataSet<SparseInstance>>();
	}

}
