// semimarkov.h
// Author: Yangfeng Ji
// Date: 09-28-2016
// Time-stamp: <yangfeng 12/12/2017 11:18:08>

#ifndef ENTITYNLM_H
#define ENTITYNLM_H

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/tensor.h"
#include "dynet/cfsm-builder.h"
#include "dynet/pretrain.h"

#include "util.h"

// #include "../beam/beamsearch.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <utility>
#include <cmath>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace dynet;


class EntityNLM {
  
private:
  LSTMBuilder builder;
  SoftmaxBuilder* smptr;
  // parameters for input
  LookupParameter p_X; // word embeddings
  
  // parameters for entity type
  Parameter p_W_R; // entity type prediction ({2, hidden_dim})
  
  // parameters for entity cluster
  Parameter p_W_E; // entity cluster prediction ({entity_dim, hidden_dim})
  Parameter p_lambda_dist;
  
  // parameters for entity length
  Parameter p_W_L; // length distribution parameter
  Parameter p_L_bias;

  // parameters for hidden -> entity transformation
  Parameter p_W_T;

  // parameter for entity embeddings
  Parameter p_E;

  // parameters for word prediction
  Parameter p_cont; // default context vector

  Parameter p_Wx;
  Parameter p_Te;
  Parameter p_Tc;
  Parameter p_Tl; // attention param for local context

  // parameters for entity embedding update
  Parameter p_W_delta; // updating weight matrix
  unsigned ntype, menlen, indim, hidim, entdim;

  // word embeddings
  unordered_map<int, vector<float>> embeddings;
  bool with_embeddings;

  // ------------------------------------
  // graph-specific variables
  // for each expression, remember to initize it in InitGraph
  Expression WR, WL, L_bias, WE, lambda_dist, WT, embed_dummy, cont_dummy, Wdelta, Wx, Te, Tc, Tl, recip_norm;
  vector<Expression> entitylist;
  vector<float> entitydist;

  map<unsigned, unsigned> map_eidx_pos; // eidx to pos in entitylist mapping
  map<unsigned, unsigned> map_pos_eidx; // pos to eidx in entitylist mapping
  // local context
  bool has_local_cont;
  vector<Expression> prev_hs; // hidden states from previous sent
  unsigned comp_method; // context composition method

  // -------------------------------------
  // string for sampled text
  string sampledtext;
  bool with_sample;

  // -------------------------------------
  // composition weights
  float lambda0, lambda1, lambda2, lambda3;


public:
  EntityNLM(){};
  EntityNLM(ParameterCollection& model,
	    unsigned vocab_size,
	    unsigned type_size,
	    unsigned men_length, // max mention length
	    Dict& d,
	    unsigned layers = 1,
	    unsigned input_dim = 32,
	    unsigned hidden_dim = 32,
	    unsigned entity_dim = 32,
	    unsigned composition_method = 0,
	    float lambda0 = 1.0,
	    float lambda1 = 1.0,
	    float lambda2 = 1.0,
	    float lambda3 = 1.0,
	    string cluster_file="",
	    string fembed="");
  
  // init a CG
  int InitGraph(ComputationGraph& cg, float drop_rate);

  // generative model
  Expression BuildGraph(const Doc& doc,
			ComputationGraph& cg,
			Dict& d,
			int err_type=0,
			float err_weight=1.0,
			int nsample=0);

  Expression BuildREGGraph(const Doc& doc,
			   ComputationGraph& cg,
			   Dict& d,
			   string& regscores);

  // discriminative model
  Expression BuildDisGraph(const Doc& doc,
			   ComputationGraph& cg);

  // get sampled text
  string get_sampledtext();

private:
  int get_index(vector<float>& vec, bool take_zero=true);

  vector<float> get_dist_feat(vector<float> entitydist, int n);
  
  int create_entity(ComputationGraph&, Expression&,
		    vector<Expression>&, vector<float>&,
		    map<unsigned, unsigned>&,
		    map<unsigned, unsigned>&,
		    int, unsigned);
  
  int update_entity(ComputationGraph&, vector<Expression>&,
		    vector<float>&, map<unsigned, unsigned>&,
		    Expression&, Expression&, Expression&,
		    Expression&, int, unsigned);

  Expression get_context(ComputationGraph& cg, Expression hidden_state, Expression local_context, Expression entity_context);

  Expression normalize_exp(ComputationGraph& cg, Expression e);

  vector<Sent> update_candidates(vector<Sent> candidates, Sent sent, int k);
};



#endif
