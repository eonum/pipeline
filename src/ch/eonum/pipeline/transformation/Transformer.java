package ch.eonum.pipeline.transformation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import ch.eonum.pipeline.core.DataPipeline;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.Parameters;

/**
 * Extracts/Transforms/Combines features.
 * @author tim
 *
 */
public class Transformer<E extends Instance> extends Parameters implements DataPipeline<E> {
	private Set<String> features;
	protected DataSet<E> dataSet;
	private List<DataPipeline<E>> inputsTraining;
	private List<DataPipeline<E>> inputsTest;
	
	public Transformer(){
		this.features = new HashSet<String>();
		this.inputsTraining = new ArrayList<DataPipeline<E>>();
		this.inputsTest = new ArrayList<DataPipeline<E>>();

	}

	/**
	 * Set the input. Depending on the concrete class of the feature extractor,
	 * the input dataset is changed and the same set is given back or a new one
	 * is created.
	 */
	public void setInputDataSet(DataSet<E> dataSet){
		this.dataSet = dataSet;
	}
	
	/**
	 * Get the output.
	 */
	public DataSet<E> getOutputDataSet(){
		return dataSet;
	}
	
	/**
	 * Get an ordered list of all features. First there are all non-numeric
	 * named features, sorted alphabetically. Then there are all numerical-named
	 * features, sorted numerically.
	 */
	public List<String> getOrderedListOfFeatures() {
		List<String> list = Arrays.asList(this.features.toArray(new String[0]));
		Collections.sort(list, new Comparator<String>(){
			@Override
			public int compare(String arg0, String arg1) {
				int i0 = -1;
				int i1 = -1;
				try {i0 = Integer.parseInt(arg0);}catch(NumberFormatException e){i0 = -1;}
				try {i1 = Integer.parseInt(arg1);}catch(NumberFormatException e){i1 = -1;}
				if(i0 == -1 && i1 == -1)
					return arg0.compareTo(arg1);
				if((i0 != -1 && i1 != -1))
					return i0 - i1;
				if(i0 == -1)
					return -1;
				return 1;
			}	
		});
		return list;
	}
	
	protected void addFeature(String feature){
		this.features.add(feature);
	}
	
	/**
	 * Transformation / extraction of the features.
	 */
	public void extract(){
		for (Instance inst : this.dataSet)
			for(String dimension : inst.features())
				this.addFeature(dimension);
	}
	
	/**
	 * Preparation of the extractor. for those extractor which need training.
	 */
	public void prepare(DataSet<E> dataset){}
	
	/** pipeline methods . **/
	@Override
	public DataSet<E> trainSystem(boolean isResultDataSetNeeded){
		DataSet<E> trainingDataSet = null;
		if(this.inputsTraining != null)
			trainingDataSet = this.inputsTraining.get(0).trainSystem(true);
		prepare(trainingDataSet);
		if(isResultDataSetNeeded)	{
			DataSet<E> temp = this.dataSet;
			this.dataSet = trainingDataSet;
			extract();
			DataSet<E> temp2 = this.getOutputDataSet();
			this.dataSet = temp;
			return temp2;
		}
		return null;
	}
	
	@Override
	public DataSet<E> testSystem(){
		if(this.inputsTest.size() != 0){
			this.resetDatasets();
			for(DataPipeline<E> each : this.inputsTest)
				this.setInputDataSet(each.testSystem());
		}
		extract();
		return this.getOutputDataSet();
	}
	
	protected void resetDatasets() {
		this.dataSet = null;
	}

	@Override
	public void addInputTraining(DataPipeline<E> input){
		this.inputsTraining.add(input);
	}
	
	@Override
	public void addInputTest(DataPipeline<E> input){
		this.inputsTest.add(input);
	}
}
