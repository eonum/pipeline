package ch.eonum.pipeline.classification.meta;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.util.FileUtil;

/**
 * Bagging ensemble with additional random selection of features (if k < 1.0)
 * (similar to the feature selection procedure used in random forests)
 * 
 * A list with all base classifiers has to be provided.
 * 
 * @author tim
 * 
 */
public class Bagging<E extends Instance> extends Classifier<E> {
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {
		PARAMETERS.put("k",
						"percentage of features which are taken into account at each component [0,1] (default 1.0)");
	}

	protected List<Classifier<E>> baseClassifiers;
	protected Random rand;
	/** only test out of bag trees when testing. */
	protected boolean oobTesting;

	public Bagging(List<Classifier<E>> baseClassifiers, Features features, int seed) {
		this.features = features;
		this.baseClassifiers = baseClassifiers;
		this.setSupportedParameters(Bagging.PARAMETERS);
		this.putParameter("k", 1.0);
		this.rand = new Random(seed);
		this.oobTesting = true;
	}

	@Override
	public void train() {
		double k = this.getDoubleParameter("k");
		int i = 0;
		for(Classifier<E> c : baseClassifiers){
			// pick features
			Features fs = getFeatures();
			if(k < 1.0){
				fs = new Features();
				for(int f = 0; f < getFeatures().size(); f++)
					if(rand.nextDouble() < k)
						fs.addFeature(getFeatures().getFeatureByIndex(f));
				fs.recalculateIndex();
			}
			c.setBaseDir(getBaseDir() + i + "/");
			FileUtil.mkdir(getBaseDir() + i + "/");
			i++;
			c.setFeatures(fs);
			DataSet<E> train = bootstrap(trainingDataSet);
			c.setTrainingSet(train);
			c.setTestSet(outOfBag(train, trainingDataSet));
		}
		
		ExecutorService service = Executors.newFixedThreadPool(Math.min(Runtime
				.getRuntime().availableProcessors(), baseClassifiers.size()));
		
		for(Classifier<E> c : baseClassifiers)
			service.submit(c);
		
		service.shutdown();
		try {
			service.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);
		} catch (InterruptedException e) {
			e.printStackTrace();
			System.exit(-1);
		}
		
	}
	
	protected <T extends Instance > DataSet<T> bootstrap(DataSet<T> data) {
		DataSet<T> bootstrap = new DataSet<T>();
		for(int i = 0; i < data.size(); i++)
			bootstrap.addInstance(data.get(rand.nextInt(data.size())));
		return bootstrap;
	}
	
	/**
	 * OOB, out of bag
	 * @param bag
	 * @param all
	 * @return
	 */
	protected <T extends Instance> DataSet<T> outOfBag(DataSet<T> bag, DataSet<T> all) {
		DataSet<T> oob = new DataSet<T>();
		for(T each : all)
			if(!bag.contains(each))
				oob.addInstance(each);
		return oob;
	}

	@Override
	public DataSet<E> test() {
		for(E e : testDataSet){
			e.putResult("result", 0.);
			e.putResult("numTrees", 0.);
			e.putResult("resultTemp", 0.);
		}
		for(Classifier<E> c : baseClassifiers){
			c.setTestSet(testDataSet);
			testDataSet = c.test();
			for(Instance each : testDataSet)
				if(!c.getTrainingDataSet().contains(each) || !this.oobTesting){
					each.putResult("resultTemp", each.getResult("resultTemp") + each.getResult("result"));
					each.putResult("numTrees", each.getResult("numTrees") + 1.0);
				}
		}
		for(Instance each : testDataSet)
			if(each.get("numTrees") != 0)
				each.putResult("result", each.getResult("resultTemp") / each.getResult("numTrees"));
		
		for(E e : testDataSet)
			e.putResult("numTrees", 0.);
		
		return this.testDataSet;
	}
}
