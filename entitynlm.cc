#include "entitynlm.h"

EntityNLM::EntityNLM(ParameterCollection& model,
		     unsigned vocab_size,
		     unsigned type_size,
		     unsigned men_length, // max mention length
		     Dict& d,
		     unsigned layers,
		     unsigned input_dim,
		     unsigned hidden_dim,
		     unsigned entity_dim,
		     unsigned composition_method,
		     float lambda0, float lambda1,
		     float lambda2, float lambda3,
		     string cluster_file,
		     string fembed) : builder(layers, input_dim, hidden_dim, model) {
  /*****************************************
   * Initialize the model
   *****************************************/
  ntype = type_size;
  menlen = men_length;
  indim = input_dim;
  hidim = hidden_dim;
  entdim = entity_dim;
  // CFSM
  if (cluster_file.size() == 0){
    //throw runtime_error("no word cluster file for CFSM");
    cerr << "Use standard softmax ..." << endl;
    smptr = new StandardSoftmaxBuilder(hidden_dim, d.size(), model, true);
  } else {
    cerr << "Use CFSM ..." << endl;
    smptr = new ClassFactoredSoftmaxBuilder(hidden_dim, cluster_file, d, model, true);
  }
  //
  with_embeddings = false;
  //
  p_X = model.add_lookup_parameters(vocab_size, {input_dim});
  p_W_R = model.add_parameters({2, hidden_dim});
  p_W_E = model.add_parameters({entity_dim, hidden_dim});
  p_lambda_dist = model.add_parameters({(unsigned)1}, 1e-6);
  p_W_L = model.add_parameters({men_length, entity_dim+hidden_dim});
  p_L_bias = model.add_parameters({men_length});
  p_W_T = model.add_parameters({entity_dim, hidden_dim});
  p_E = model.add_parameters({entity_dim});
  p_cont = model.add_parameters({entity_dim});
  p_W_delta = model.add_parameters({entity_dim, hidden_dim});
  p_Wx = model.add_parameters({vocab_size, hidden_dim});
  p_Te = model.add_parameters({hidden_dim, entity_dim});
  p_Tc = model.add_parameters({hidden_dim, entity_dim});
  p_Tl = model.add_parameters({hidden_dim, hidden_dim});
  //
  sampledtext = "";
  with_sample = false;
  has_local_cont = false;
  comp_method = composition_method;
}

int EntityNLM::InitGraph(ComputationGraph& cg, float drop_rate){
  /*****************************************
   * Should be used per doc to empty the graph
   *****************************************/
  if (drop_rate > 0){
    builder.set_dropout(drop_rate);
  } else {
    builder.disable_dropout();
  }
  builder.new_graph(cg);
  smptr->new_graph(cg);
  // TODO: add back distance based features
  // vector<float> doc_dist(n_entity, 0.0), sent_dist;
  // expressions
  WR = parameter(cg, p_W_R);
  WL = parameter(cg, p_W_L);
  L_bias = parameter(cg, p_L_bias);
  WE = parameter(cg, p_W_E);
  lambda_dist = parameter(cg, p_lambda_dist);
  WT = parameter(cg, p_W_T);
  embed_dummy = parameter(cg, p_E);
  cont_dummy = parameter(cg, p_cont);
  Wdelta = parameter(cg, p_W_delta);
  Wx = parameter(cg, p_Wx);
  Te = parameter(cg, p_Te);
  Tc = parameter(cg, p_Tc);
  Tl = parameter(cg, p_Tl);
  if (drop_rate > 0){
    WR = dropout(WR, drop_rate);
    WL = dropout(WL, drop_rate);
    WE = dropout(WE, drop_rate);
    WT = dropout(WT, drop_rate);
    Wx = dropout(Wx, drop_rate);
    Te = dropout(Te, drop_rate);
    Tc = dropout(Tc, drop_rate);
    Tl = dropout(Tl, drop_rate);
  }
  //
  entitylist.clear(); // = vector<Expression>();
  entitydist.clear(); // = vector<float>();
  // entitylist.push_back(normalize_exp(cg, embed_dummy));
  // entitylist.push_back(logistic(embed_dummy));
  entitylist.push_back(embed_dummy);
  entitydist.push_back(0.0);
  map_eidx_pos.clear();
  map_pos_eidx.clear();
  //
  has_local_cont = false;
  prev_hs.clear();
  // 
  return 0;
}

Expression EntityNLM::BuildGraph(const Doc& doc,
				 ComputationGraph& cg,
				 Dict& d,
				 int err_type,
				 float err_weight,
				 int nsample){
  /********************************************
   * Build a CG per doc
   ********************************************/
  
  // Q: how to check whether a CG has been initialized?

  // for each new doc, reset global variables
  with_sample = nsample;
  //
  int closest_eidx = -1;
  // index variable
  map<unsigned, unsigned>::iterator itc, itn;
  // build the coref graph and LM for a given doc
  vector<Expression> t_errs, e_errs, l_errs, x_errs;
  const unsigned nsent = doc.sents.size(); // doc length
  // get the dummy context vector
  Expression prev_cont_mat;
  Expression cont = cont_dummy; // normalize_exp(cg, cont_dummy);
  Expression x_t, h_t;
  for (unsigned n = 0; n < nsent; n++){
    builder.start_new_sequence();
    if (prev_hs.size() > 0){
      has_local_cont = true;
      // cerr << "prev_hs.size() = " << prev_hs.size() << endl;
      prev_cont_mat = concatenate_cols(prev_hs);
      prev_hs.clear(); // clean up
    } 
    auto& sent = doc.sents[n]; // get the current sentence
    unsigned nword = sent.size() - 1; // sent length
    // cerr << "nword = " << nword << endl;
    for (unsigned t = 0; t < nword; t++){
      // get mention type (if there is one)
      auto& curr_tt = sent[t].tidx;
      auto& curr_xt = sent[t].xidx;
      auto& curr_et = sent[t].eidx;
      auto& curr_lt = sent[t].mlen;
      auto& next_tt = sent[t+1].tidx;
      auto& next_xt = sent[t+1].xidx;
      auto& next_et = sent[t+1].eidx;
      auto& next_lt = sent[t+1].mlen;
      // update closest_eidx
      if (curr_tt > 0){
	closest_eidx = curr_et;
      }
      // add current token onto CG
      x_t = lookup(cg, p_X, curr_xt);
      // if (drop_rate > 0) x_t = dropout(x_t, drop_rate);
      
      // get hidden state h_t
      h_t = builder.add_input(x_t);
      // normalize hidden state
      // h_t = normalize_exp(cg, h_t);
      // 
      prev_hs.push_back(h_t); // keep it for local context
      
      // ---------------------------------------------
      // update the entity embedding at the end of the mention
      if ((curr_tt > 0) and (curr_et > 0)){
	// Expression entrep, recip_norm;
	itc = map_eidx_pos.find(curr_et);
	if (itc == map_eidx_pos.end()){
	  // create a new entity
	  create_entity(cg, embed_dummy, entitylist, entitydist,
			map_eidx_pos, map_pos_eidx, curr_et, n);
	}
	// based on comtextual information, update entity embedding
        // cerr << "update entity embedding" << endl;
	update_entity(cg, entitylist, entitydist,
		      map_eidx_pos, h_t, Wdelta, WT,
		      cont, curr_et, n);
      }
      
      if (curr_lt == 1){
	// ---------------------------------------------
	// next entity type prediction
	Expression t_logit = (WR * h_t);
	t_errs.push_back(pickneglogsoftmax(t_logit, next_tt));
	// ---------------------------------------------
	// entity prediction
	if (next_tt > 0){
	  // get distance feature
	  // vector<float> feat_dist;
	  // for (auto& val : entitydist){
	  //   feat_dist.push_back(val-n);
	  // }
	  vector<float> feat_dist = get_dist_feat(entitydist, n);
	  //
	  Expression entmat = concatenate_cols(entitylist);
	  Expression e_logit = ((transpose(entmat) * WE) * h_t) +
	    exp(input(cg, {(unsigned)feat_dist.size()}, feat_dist) * lambda_dist);
	  Expression e_err;
	  itn = map_eidx_pos.find(next_et);
	  if (itn != map_eidx_pos.end()){
	    // if this is not a new entity
	    e_err = pickneglogsoftmax(e_logit, itn->second);
	  } else {
	    // if this is a new entity
	    e_err = pickneglogsoftmax(e_logit, (unsigned)0);
	  }
	  // float v_e_err = as_scalar(cg.incremental_forward(e_err));
	  e_errs.push_back(e_err);
	}
	
	// ---------------------------------------------
	// entity length prediction
	if (next_et > 0){
	  Expression l_logit;
	  itn = map_eidx_pos.find(next_et);
	  if (itn != map_eidx_pos.end()){
	    l_logit = WL * concatenate({h_t, entitylist[itn->second]}) + L_bias;
	  } else {
	    l_logit = WL * concatenate({h_t, entitylist[0]}) + L_bias;
	  }
	  l_errs.push_back(pickneglogsoftmax(l_logit, next_lt-1));
	}
      }
      
      // -----------------------------------------------
      // word prediction

      // construct local context
      if (has_local_cont){
      	Expression alpha = softmax((transpose(prev_cont_mat) * Tl) * h_t);
      	cont = prev_cont_mat * alpha;
	// normalize local context
	// cont = normalize_exp(cg, cont);
      }
      // need to refine this part about incorporating
      // different sources of context
      Expression x_err, w_logit, entity_cont;
      if (next_tt > 0){
	itn = map_eidx_pos.find(next_et);
	if (itn != map_eidx_pos.end()){
	  // entity_cont = Te * entitylist[itn->second];
	  entity_cont = entitylist[itn->second];
	} else {
	  // entity_cont = Te * entitylist[0];
	  entity_cont = entitylist[0];
	}
      } else {
	if (closest_eidx > 0){
	  itn = map_eidx_pos.find(closest_eidx);
	  // entity_cont = Te * entitylist[itn->second];
	  entity_cont = entitylist[itn->second];
	} else {
	  switch(comp_method){
	  case 2:
	  case 3:
	  case 5:
	    entity_cont = ones(cg, {hidim});
	    // cerr << "ones" << endl;
	    break;
	  default:
	    entity_cont = zeros(cg, {hidim});
	    // cerr << "zeros" << endl;
	    break;
	  }
	}
      }
      // float h_s = as_scalar(cg.incremental_forward(squared_norm(h_t)));
      // float c_s = as_scalar(cg.incremental_forward(squared_norm(cont)));
      // float e_s = as_scalar(cg.incremental_forward(squared_norm(entity_cont)));
      // cerr << "h_s = " << h_s << " c_s = " << c_s
      // 	   << " e_s = " << e_s << endl;
      w_logit = get_context(cg, h_t, cont, entity_cont);
      x_err = smptr->neg_log_softmax(w_logit, next_xt);
      x_errs.push_back(x_err);
    } // end of sentence
    // cont = h_t; // switch to the last sentence as context
  } // end of document

  // **************************************************
  // generation function
  int thresh = 30, counter = 0;
  int xSOS = d.convert("<s>");
  int xEOS = d.convert("</s>");
  int xUNK = d.convert("UNK");
  ostringstream oss;
  // construct local context mat
  if (prev_hs.size() > 0){
    prev_cont_mat = concatenate_cols(prev_hs);
    prev_hs.clear();
    has_local_cont = true;
  } else {
    throw runtime_error("no local context !!!");
  }
  // create a copy about context
  // anything else?
  Expression cont_copy = cont;
  vector<Expression> entitylist_copy = entitylist;
  vector<float>& entitydist_copy = entitydist;
  int closest_eidx_copy = closest_eidx;
  // sample counter
  int niter = 0;
  oss << "\n";
  while (niter < nsample){
    oss << "Sample " << niter << " = ";
    niter ++;
    // --------------------------------
    // initialization
    int curr_tt = 0;
    int curr_et = 0;
    int curr_lt = 1;
    int curr_xt = xSOS;
    int next_tt, next_et, next_lt, next_xt;
    counter = 0; // token counter
    // load contextual information back
    cont = cont_copy;
    entitylist = entitylist_copy;
    entitydist = entitydist_copy;
    closest_eidx = closest_eidx_copy;
    // -------------------------------
    // create a new sequence
    builder.start_new_sequence();
    // start sampling
    while ((bool)nsample){
      // update closest_eidx
      if (curr_tt > 0){
	closest_eidx = curr_et;
      }
      x_t = lookup(cg, p_X, curr_xt);
      h_t = builder.add_input(x_t);
      // update entity
      if (curr_tt > 0){
	update_entity(cg, entitylist, entitydist,
		      map_eidx_pos, h_t, Wdelta,
		      WT, cont, curr_et,
		      doc.sents.size());
	// need to reload the original state back
	// after one pass generation
      }
      // 
      if (curr_lt <= 1){ // update next_tt | it cannot be less than 1, but just in case
	// sample entity type
	Expression t_prob = softmax(WR * h_t);
	vector<float> vt_prob = as_vector(cg.incremental_forward(t_prob));
	next_tt = get_index(vt_prob, true);
	if (next_tt > 0){ // sample an entity
	  vector<float> feat_dist = get_dist_feat(entitydist, doc.sents.size());
	  Expression entmat = concatenate_cols(entitylist);
	  Expression e_prob = softmax(((transpose(entmat) * WE) * h_t) +
				      exp(input(cg, {(unsigned)feat_dist.size()}, feat_dist) * lambda_dist));
	  vector<float> ve_prob = as_vector(cg.incremental_forward(e_prob));
	  // this line can be modified to avoid generating new entities
	  unsigned next_et_pos = get_index(ve_prob, false);
	  // now check whether this is a new entity
	  if (next_et_pos == 0){ // if it is new
	    cerr << "create a new entity ... " << endl;
	    throw runtime_error("creating new entity is foridden in sampling");
	  }
	  itn = map_pos_eidx.find(next_et_pos);
	  next_et = itn->second;
	  // sample the length for new entity
	  Expression l_prob = softmax(WL * concatenate({h_t, entitylist[next_et_pos]}) + L_bias);
	  vector<float> vl_prob = as_vector(cg.incremental_forward(l_prob));
	  next_lt = get_index(vl_prob, true) + 1;
	} else { // a content word
	  next_et = 0;
	  next_lt = 1;
	}
      } else { // previous entity info
	next_tt = curr_tt;
	next_et = curr_et;
	next_lt = curr_lt - 1;
      }
      
      // construct local context vector
      if (has_local_cont){
	Expression alpha = softmax((transpose(prev_cont_mat) * Tl) * h_t);
	cont = prev_cont_mat * alpha;
      }
      
      // now based on next_tt, decide what to do
      Expression w_logit, entity_cont;
      if (next_tt > 0){ // within an entity mention
	// get the pos of entity in the entity list
	itn = map_eidx_pos.find(next_et); 
      } else { // just a content word
	itn = map_eidx_pos.find(closest_eidx);
      }
      entity_cont = entitylist[itn->second];
      w_logit = get_context(cg, h_t, cont, entity_cont);
      next_xt = smptr->sample(w_logit);
      // Expression w_prob = smptr->full_log_distribution(w_logit);
      // vector<float> vw_prob = as_vector(cg.incremental_forward(w_prob));
      // next_xt = get_index(vw_prob, true);
      while (next_xt == xUNK){
	// !!!!!! For now, only works for random sampling
	// next_xt = get_index(vw_prob, true);
	next_xt = smptr->sample(w_logit);
      }
      oss << d.convert(next_xt) << "|" << next_tt
	  << "|" << next_et << "|" << next_lt << " ";
      
      // extra constraint on sampling
      // while ((next_xt == xEOS) and (next_tt > 0)){
      //   // please don't stop in the middle of a mention
      //   next_xt = smptr->sample(w_logit);
      // }
      
      // update curr_*
      curr_tt = next_tt;
      curr_et = next_et;
      curr_lt = next_lt;
      curr_xt = next_xt;
      // Token newtoken = {next_tt, next_et, next_xt, next_lt};
      // sampled_sent.push_back(newtoken);
      counter ++;
      if ((curr_xt == xEOS) or (counter >= thresh)){
	break;
      }
    } // a single sample
    cerr << "length of sampled text = " << counter << endl;
    oss << "\n";
    // rewind back to the state before generation
    for (int i = 0; i < counter; i++){
      builder.rewind_one_step();
    }
  } // end of sampling
  
  if ((bool)nsample){
    sampledtext = oss.str();
  }

  // **************************************************
  // return errors
  Expression i_nerr;
  if (err_type == 0){
    // full errors
    if (e_errs.size() > 0){
      i_nerr = ((sum(t_errs) + sum(e_errs) + sum(l_errs)) * err_weight) + sum(x_errs);
    } else {
      // doesn't have any entity
      i_nerr = sum(t_errs) + sum(x_errs);
    }
  } else if (err_type == 1){
    // word prediction errors
    // cerr << "err_type = " << err_type << endl;
    i_nerr = sum(x_errs);
  } else {
    abort();
  }
  // finally, return the loss
  return i_nerr;
}

/********************************************************
 *
 ********************************************************/
Expression EntityNLM::BuildREGGraph(const Doc& doc,
				    ComputationGraph& cg,
				    Dict& d,
				    string& regscores){
  regscores.clear();
  // for each new doc, reset global variables
  int closest_eidx = -1;
  // index variable
  map<unsigned, unsigned>::iterator itc, itn;
  // build the coref graph and LM for a given doc
  vector<Expression> t_errs, e_errs, l_errs, x_errs;
  const unsigned nsent = doc.sents.size(); // doc length
  // get the dummy context vector
  Expression prev_cont_mat;
  Expression cont = cont_dummy; // normalize_exp(cg, cont_dummy);
  Expression x_t, h_t;
  // for REG
  vector<Sent> candidates;
  ostringstream oss;
  oss << "\n";
  
  for (unsigned n = 0; n < nsent; n++){
    builder.start_new_sequence();
    if (prev_hs.size() > 0){
      has_local_cont = true;
      prev_cont_mat = concatenate_cols(prev_hs);
      prev_hs.clear(); // clean up
    } 
    auto& sent = doc.sents[n]; // get the current sentence
    unsigned nword = sent.size() - 1; // sent length

    for (unsigned t = 0; t < nword; t++){

      // get mention type (if there is one)
      auto& curr_tt = sent[t].tidx;
      auto& curr_xt = sent[t].xidx;
      auto& curr_et = sent[t].eidx;
      auto& curr_lt = sent[t].mlen;
      // update closest_eidx
      if (curr_tt > 0){
	closest_eidx = curr_et;
      }
      // add current token onto CG
      x_t = lookup(cg, p_X, curr_xt);      
      // get hidden state h_t
      h_t = builder.add_input(x_t);
      prev_hs.push_back(h_t); // keep it for local context
      
      // get information about next step
      Expression copy_x_t = x_t;
      Expression copy_h_t = h_t;
      auto& next_tt = sent[t+1].tidx;
      if ((curr_lt == 1) and (next_tt > 0)){
	// update candidate with current mention
	candidates = update_candidates(candidates, sent, t+1);
	// candidate mention evaluation
	// do something dirty and reload the original state back after it's done
	for (auto& phrase : candidates){
	  // for each candidate phrase

	  // OK, we already know the next few tokens belong to
	  // an entity mention, let's see whether our model
	  // could figure out which one

	  // ---------------------
	  // entity prediction
	  auto& next_et = phrase[0].eidx;
	  vector<float> feat_dist = get_dist_feat(entitydist, n);
	  Expression entmat = concatenate_cols(entitylist);
	  Expression e_logit = ((transpose(entmat) * WE) * h_t)
	    + exp(input(cg, {(unsigned)feat_dist.size()},
			feat_dist) * lambda_dist);
	  Expression e_err;
	  itn = map_eidx_pos.find(next_et);
	  if (itn != map_eidx_pos.end()){
	    e_err = pickneglogsoftmax(e_logit, itn->second);
	  } else {
	    // ok, this is a new entity
	    e_err = pickneglogsoftmax(e_logit, (unsigned)0);
	  }
	  // -----------------------
	  // entity length prediction
	  auto& next_lt = phrase[0].mlen;
	  Expression l_logit;
	  itn = map_eidx_pos.find(next_et);
	  if (itn != map_eidx_pos.end()){
	    l_logit = WL * concatenate({h_t, entitylist[itn->second]}) + L_bias;
	  } else {
	    l_logit = WL * concatenate({h_t, entitylist[0]}) + L_bias;
	  }
	  Expression l_err = pickneglogsoftmax(l_logit, next_lt-1);
	  // -----------------------
	  // entity mention prediction
	  Expression w_logit, entity_cont;
	  Expression x_err = zeros(cg, {1});
	  if (has_local_cont){
	    Expression alpha = softmax((transpose(prev_cont_mat) * Tl) * h_t);
	    cont = prev_cont_mat * alpha;
	  }
	  oss << next_et << "_";
	  for (auto& tok : phrase){
	    auto& next_xt = tok.xidx;
	    // get the actual token
	    oss << d.convert(next_xt) << "_";
	    // 
	    itn = map_eidx_pos.find(next_et);
	    if (itn != map_eidx_pos.end()){
	      entity_cont = entitylist[itn->second];
	    } else {
	      // new entity
	      entity_cont = entitylist[0];
	    }
	    w_logit = get_context(cg, h_t, cont, entity_cont);
	    x_err = x_err + smptr->neg_log_softmax(w_logit,
						   next_xt);
	    // update the hidden statement
	    x_t = lookup(cg, p_X, next_xt);
	    h_t = builder.add_input(x_t);
	  }
	  int psize = phrase.size();
	  x_err = x_err / psize;
	  // rewind the RNN builder
	  for (int k = 0; k < psize; k ++){
	    builder.rewind_one_step();
	  }
	  Expression score = e_err + l_err + x_err;
	  oss << ":" << as_scalar(cg.incremental_forward(score)) << "\t";
	  // reset the input and hidden states, before computing next candidate
	  x_t = copy_x_t;
	  h_t = copy_h_t;
	} // end for each candidate
	oss << "\n";
      }
      // just in case something goes wrong, which I don't think
      // it will happen
      x_t = copy_x_t;
      h_t = copy_h_t;

      // *************************************************
      // now, let's continue the CG with ground truth
      auto& next_xt = sent[t+1].xidx;
      auto& next_et = sent[t+1].eidx;
      auto& next_lt = sent[t+1].mlen;
      // ---------------------------------------------
      // update the entity embedding at the end of the mention
      if ((curr_tt > 0) and (curr_et > 0)){
	// Expression entrep, recip_norm;
	itc = map_eidx_pos.find(curr_et);
	if (itc == map_eidx_pos.end()){
	  // create a new entity
	  create_entity(cg, embed_dummy, entitylist, entitydist,
			map_eidx_pos, map_pos_eidx, curr_et, n);
	}
	// based on comtextual information, update entity embedding
        // cerr << "update entity embedding" << endl;
	update_entity(cg, entitylist, entitydist,
		      map_eidx_pos, h_t, Wdelta, WT,
		      cont, curr_et, n);
      }
      
      if (curr_lt == 1){
	// ---------------------------------------------
	// next entity type prediction
	Expression t_logit = (WR * h_t);
	t_errs.push_back(pickneglogsoftmax(t_logit, next_tt));
	// ---------------------------------------------
	// entity prediction
	if (next_tt > 0){
	  // get distance feature
	  vector<float> feat_dist = get_dist_feat(entitydist, n);
	  //
	  Expression entmat = concatenate_cols(entitylist);
	  Expression e_logit = ((transpose(entmat) * WE) * h_t) +
	    exp(input(cg, {(unsigned)feat_dist.size()}, feat_dist) * lambda_dist);
	  Expression e_err;
	  itn = map_eidx_pos.find(next_et);
	  if (itn != map_eidx_pos.end()){
	    // if this is not a new entity
	    e_err = pickneglogsoftmax(e_logit, itn->second);
	  } else {
	    // if this is a new entity
	    e_err = pickneglogsoftmax(e_logit, (unsigned)0);
	  }
	  // float v_e_err = as_scalar(cg.incremental_forward(e_err));
	  e_errs.push_back(e_err);
	}
	
	// ---------------------------------------------
	// entity length prediction
	if (next_et > 0){
	  Expression l_logit;
	  itn = map_eidx_pos.find(next_et);
	  if (itn != map_eidx_pos.end()){
	    l_logit = WL * concatenate({h_t, entitylist[itn->second]}) + L_bias;
	  } else {
	    l_logit = WL * concatenate({h_t, entitylist[0]}) + L_bias;
	  }
	  l_errs.push_back(pickneglogsoftmax(l_logit, next_lt-1));
	}
      }
      
      // -----------------------------------------------
      // word prediction

      // construct local context
      if (has_local_cont){
      	Expression alpha = softmax((transpose(prev_cont_mat) * Tl) * h_t);
      	cont = prev_cont_mat * alpha;
	// normalize local context
	// cont = normalize_exp(cg, cont);
      }
      // need to refine this part about incorporating
      // different sources of context
      Expression x_err, w_logit, entity_cont;
      if (next_tt > 0){
	itn = map_eidx_pos.find(next_et);
	if (itn != map_eidx_pos.end()){
	  // entity_cont = Te * entitylist[itn->second];
	  entity_cont = entitylist[itn->second];
	} else {
	  // entity_cont = Te * entitylist[0];
	  entity_cont = entitylist[0];
	}
      } else {
	if (closest_eidx > 0){
	  itn = map_eidx_pos.find(closest_eidx);
	  // entity_cont = Te * entitylist[itn->second];
	  entity_cont = entitylist[itn->second];
	} else {
	  switch(comp_method){
	  case 2:
	  case 3:
	  case 5:
	    entity_cont = ones(cg, {hidim});
	    // cerr << "ones" << endl;
	    break;
	  default:
	    entity_cont = zeros(cg, {hidim});
	    // cerr << "zeros" << endl;
	    break;
	  }
	}
      }

      w_logit = get_context(cg, h_t, cont, entity_cont);
      x_err = smptr->neg_log_softmax(w_logit, next_xt);
      x_errs.push_back(x_err);
    } // end of sentence
    // cont = h_t; // switch to the last sentence as context
  } // end of document

  // get scores
  regscores = oss.str();

  // 
  Expression i_nerr = sum(t_errs) + sum(e_errs) + sum(l_errs) + sum(x_errs);
  // finally, return the loss
  return i_nerr;
}

/***************************************************************
 *
 ***************************************************************/

Expression EntityNLM::BuildDisGraph(const Doc& doc,
				    ComputationGraph& cg){
  //
  map<unsigned, unsigned>::iterator itc, itn;
  //
  vector<Expression> t_errs, e_errs, l_errs;
  const unsigned nsent = doc.sents.size();
  //
  Expression x_t, h_t;
  int prev_tt, prev_et, prev_lt, prev_xt;
  int curr_tt, curr_et, curr_lt, curr_xt;
  for (unsigned n = 0; n < nsent; n++){
    builder.start_new_sequence();
    auto& sent = doc.sents[n];
    unsigned nword = sent.size();
    for (unsigned t = 1; t < nword; t++){
      // cerr << "n = " << n << " t = " << t << endl;
      prev_tt = sent[t-1].tidx;
      prev_et = sent[t-1].eidx;
      prev_lt = sent[t-1].mlen;
      prev_xt = sent[t-1].xidx;
      curr_tt = sent[t].tidx;
      curr_et = sent[t].eidx;
      curr_lt = sent[t].mlen;
      curr_xt = sent[t].xidx;
      // cerr << "prev = " << prev_tt << " " << prev_et << " " << prev_lt << " " << prev_xt
      // 	   << " curr = " << curr_tt << " " << curr_et << " " << curr_lt << " " << curr_xt << endl;
      // feed the current word
      x_t = lookup(cg, p_X, curr_xt);
      h_t = builder.add_input(x_t);
      //
      if (prev_lt == 1){
	Expression t_logit = (WR * h_t);
	t_errs.push_back(pickneglogsoftmax(t_logit, curr_tt));
	// --------------------------------------
	// entity prediction
	if (curr_tt > 0){
	  Expression entmat = concatenate_cols(entitylist);
	  Expression e_logit = (transpose(entmat) * WE) * h_t;
	  Expression e_err;
	  itn = map_eidx_pos.find(curr_et);
	  if (itn != map_eidx_pos.end()){
	    // this an existing new entity
	    e_err = pickneglogsoftmax(e_logit, itn->second);
	  } else {
	    // this is a new entity
	    e_err = pickneglogsoftmax(e_logit, (unsigned)0);
	    // create the entity embedding
	    create_entity(cg, embed_dummy, entitylist, entitydist,
			  map_eidx_pos, map_pos_eidx, curr_et, n);
	  }
	  e_errs.push_back(e_err);
	}
	if (curr_et > 0){
	  Expression l_logit;
	  itn = map_eidx_pos.find(curr_et);
	  if (itn != map_eidx_pos.end()){
	    l_logit = WL * concatenate({h_t, entitylist[itn->second]}) + L_bias;
	  } else {
	    l_logit = WL * concatenate({h_t, entitylist[0]}) + L_bias;
	  }
	  l_errs.push_back(pickneglogsoftmax(l_logit, curr_lt - 1));
	}
	// update entity embedding
	// not necessary to use another if statement,
	// but it is more readable this way
	if (curr_et > 0){
	  itc = map_eidx_pos.find(curr_et);
	  if (itc == map_eidx_pos.end()){
	    // create a new entity
	    create_entity(cg, embed_dummy, entitylist, entitydist,
			  map_eidx_pos, map_pos_eidx, curr_et, n);
	  }
	}
      } // prev_lt = 1
    } // word
  } // sentence
  // cerr << "t_errs.size() = " << t_errs.size() << endl;
  Expression errs;
  if (t_errs.size() > 0){
    errs = sum(t_errs) + sum(e_errs) + sum(l_errs);
  } else {
    cerr << "zero coreference information here" << endl;
    errs = zeros(cg, {1});
  }
  // cerr << "return ..." << endl;
  return errs;
}

string EntityNLM::get_sampledtext(){
  if (not with_sample){
    sampledtext = "";
  }
  return sampledtext;
}

int EntityNLM::get_index(vector<float>& vec, bool take_zero){
  bool greedy = false; // preset decoding method
  int val;
  // cerr << "vec = ";
  // for (auto& val : vec) cerr << val << " ";
  // cerr << endl;
  // cerr << "take zero = " << take_zero << endl;
  if (greedy){
    if (take_zero){
      val = argmax(vec);
    } else {
      // slice the vec to get rid of the first element
      vector<float>::const_iterator first = vec.begin() + 1;
      vector<float>::const_iterator last = vec.end();
      vector<float> new_vec(first, last);
      // find the secondary best
      val = argmax(new_vec) + 1;
    }
    // cerr << "max val = " << val << endl;
  } else {
    val = sample_dist(vec);

    if (!take_zero){
      while (val == 0){
	// sample again
	val = sample_dist(vec);
      }
    }
    // cerr << "sample val = " << val << endl;
  }

  return val;
}

int EntityNLM::create_entity(ComputationGraph& cg,
			     Expression& embed_dummy,
			     vector<Expression>& entitylist,
			     vector<float>& entitydist,
			     map<unsigned, unsigned>& map_eidx_pos,
			     map<unsigned, unsigned>& map_pos_eidx,
			     int curr_eidx,
			     unsigned nsent){
  map_eidx_pos[curr_eidx] = entitylist.size();
  map_pos_eidx[entitylist.size()] = curr_eidx;
  Expression entrep, recip_norm;
  entrep = random_normal(cg, {entdim}) * 1e-2;
  entrep = embed_dummy + entrep;
  // entrep = logistic(embed_dummy + entrep);
  // entrep = normalize_exp(cg, entrep);
  entitylist.push_back(entrep);
  entitydist.push_back(nsent);
  return (entitylist.size() - 1);
}

int EntityNLM::update_entity(ComputationGraph& cg,
			     vector<Expression>& entitylist,
			     vector<float>& entitydist,
			     map<unsigned, unsigned>& map_eidx_pos,
			     Expression& h_t, Expression& Wdelta,
			     Expression& WT, Expression& cont,
			     int curr_eidx, unsigned nsent){
  map<unsigned, unsigned>::iterator it = map_eidx_pos.find(curr_eidx);
  Expression entrep, recip_norm, delta;
  entrep = entitylist[it->second];
  delta = logistic((transpose(entrep) * Wdelta) * h_t);
  entrep = (entrep * (1 - delta)) + ((WT * h_t) * delta);
  // entitylist[it->second] = normalize_exp(cg, entrep);
  // entitylist[it->second] = logistic(entrep);
  entitylist[it->second] = entrep;
  entitydist[it->second] = nsent; // update distance feature
  cont = entitylist[it->second];
  return 0;
}

vector<float> EntityNLM::get_dist_feat(vector<float> entitydist, int n){
  vector<float> feat_dist;
  for (auto& val : entitydist){
    feat_dist.push_back(val - n);
  }
  return feat_dist;
}


Expression EntityNLM::get_context(ComputationGraph& cg, Expression hidden_state, Expression local_context, Expression entity_context){
  Expression summary_context, tmp;
  switch (comp_method){
  case 0:
    // add
    summary_context = hidden_state + local_context + entity_context;
    break;
  case 1:
    // something Yangfeng made up
    summary_context = cmult(hidden_state, local_context) + entity_context;
    break;
  case 2:
    // element-wise mult
    summary_context = cmult(cmult(hidden_state, local_context),entity_context);
    break;
  case 3:
    // polynomial
    summary_context = lambda0 * cmult(hidden_state, local_context) + lambda1 * cmult(local_context, entity_context) + lambda2 * cmult(hidden_state, entity_context) + lambda3 * cmult(hidden_state, cmult(local_context, entity_context));
    break;
  case 4:
    // arithmetic mean
    // no difference from method 0
    summary_context = (hidden_state + local_context + entity_context)/3;
    break;
  case 5:
    // approximate geometric mean
    summary_context = cmult(cmult(hidden_state, local_context), entity_context);
    summary_context = pow(abs(summary_context), input(cg, 0.33));
    break;
  case 6:
    // max-pooling
    tmp = max(hidden_state, local_context);
    summary_context = max(tmp, entity_context);
    break;
  case 7:
    // min-pooling
    tmp = min(hidden_state, local_context);
    summary_context = min(tmp, entity_context);
    break;
  case 8:
    // baseline 1: hidden state only
    summary_context = hidden_state;
    break;
  case 9:
    // baseline 2: without entity rep
    summary_context = max(hidden_state, local_context);
    break;
  case 10:
    // baseline 3: without local context
    summary_context = max(hidden_state, entity_context);
    break;
  default:
    throw runtime_error("Unknown composition method");
  }
  return summary_context;
}

Expression EntityNLM::normalize_exp(ComputationGraph& cg,
				    Expression e){
  Expression recip_norm, normalized_e, sqnorm_e;
  recip_norm = pow(squared_norm(e), input(cg, -0.5));
  normalized_e = e * recip_norm;
  return normalized_e;
}

vector<Sent> EntityNLM::update_candidates(vector<Sent> candidates, Sent sent, int k){
  Sent phrase;
  for (int i = 0; i < sent[k].mlen; i ++)
    phrase.push_back(sent[k+i]);
  candidates.push_back(phrase);
  return candidates;
}
