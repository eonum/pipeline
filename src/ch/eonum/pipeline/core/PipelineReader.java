package ch.eonum.pipeline.core;

import java.io.File;

/**
 * Read a data set in the pipeline format. Produces SparseInstances.
 * @author tim
 *
 */
public class PipelineReader extends DataSetReader<SparseInstance> {

	public PipelineReader(String fileName) {
		super(fileName);
	}

	public PipelineReader(File f) {
		super(f);
	}

	@Override
	protected void convertLine(DataSet<SparseInstance> data, String line) {
		data.addInstance(SparseInstance.parse(line));
	}

}
