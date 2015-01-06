package ch.eonum.pipeline.transformation;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.HashMap;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.SparseInstance;

/**
 * Normalize a data set to have a mean = 0.0 and a standard deviation = 1.0 for
 * each feature.
 * 
 * @author tim
 * 
 */
public class StandardNormalizer<E extends Instance> extends Transformer<E> {
	private SparseInstance mean;
	private SparseInstance standardDeviation;

	/**
	 * Create a normalizer from a given data set. Mean and standard deviation
	 * are calculated out of the data set.
	 * 
	 * @param dataset
	 */
	public StandardNormalizer(DataSet<E> dataset) {
		this.prepare(dataset);
	}
	
	@Override
	public void prepare(DataSet<E> dataset){
		mean = new SparseInstance("", "", new HashMap<String, Double>());
		standardDeviation = new SparseInstance("", "", new HashMap<String, Double>());
		for(Instance each : dataset)
			mean.add(each);
		mean.divideBy(dataset.size());
		for(Instance each : dataset){
			each = mean.minusStateless(each);
			standardDeviation.add(each.timesStateless(each));
		}
		standardDeviation.divideBy(dataset.size());
		standardDeviation.sqrt();
	}
	
	/**
	 * Create a normalizer by loading mean and standard deviation from file.
	 * format: one line for each feature: 'feature meanvalue sdValue'
	 * @param fileName
	 */
	public StandardNormalizer(String fileName) {
		mean = new SparseInstance("", "", new HashMap<String, Double>());
		standardDeviation = new SparseInstance("", "", new HashMap<String, Double>());
		try {
			FileInputStream fstream = new FileInputStream(fileName);
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String strLine;
			
			while ((strLine = br.readLine()) != null) {
				String[] values = strLine.split(" ");
				mean.put(values[0], Double.valueOf(values[1]));
				standardDeviation.put(values[0], Double.valueOf(values[2]));
			}
			in.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public void extract(){
		super.extract();
		for(Instance each : this.dataSet){
			each.minus(mean);
			each.divideBy(standardDeviation);
		}
	}

	public void setMean(SparseInstance mean) {
		this.mean = mean;
	}

	public Instance getMean() {
		return mean;
	}

	public void setStandardDeviation(SparseInstance standardDeviation) {
		this.standardDeviation = standardDeviation;
	}

	public Instance getStandardDeviation() {
		return standardDeviation;
	}

	public void writeToFile(String fileName) {
		try {
			PrintStream p = new PrintStream(new FileOutputStream(fileName));
			for(String feature : mean.features())
				p.println(feature + " " + mean.get(feature) + " " + standardDeviation.get(feature));
			p.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
