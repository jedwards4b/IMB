////
/// @file shr_spmd.c
/// @author Algorithms modeled after spmd_utils in the Community Atmosphere Model; C translation Jim Edwards
/// @date 2014
/// @brief MPI_Gather, MPI_Gatherv, and MPI_Alltoallw with flow control options
///
///

#include "shr_spmd.h"

int maxreq=MAX_GATHER_BLOCK_SIZE;
/**
 ** @brief Used to sort map points in the shr_create_comm_datatypes
*/
typedef struct mapsort
{
  int rfrom;
  MPI_Offset soffset;
  MPI_Offset gmap;
} mapsort;

///
/// Wrapper for MPI calls to print the Error string on error
///
int shr_spmd_CheckMPIReturn(const int ierr,const char file[],const int line)
{

  if(ierr != MPI_SUCCESS){
    char errstring[MPI_MAX_ERROR_STRING];
    int errstrlen;
    int mpierr = MPI_Error_string( ierr, errstring, &errstrlen);

    fprintf(stderr, "MPI ERROR: %s in file %s at line %d\n",errstring, file, line);
    return ierr;
  }
  return SHR_NOERR;
}

/**
 ** @internal
 ** compare offsets is used by the sort in the shr_create_comm_datatypes
 ** @endinternal
 */
int compare_offsets(const void *a,const void *b)
{
  mapsort *x = (mapsort *) a;
  mapsort *y = (mapsort *) b;
  return (int) (x->gmap - y->gmap);
}


///
///   Given a map on one or more of the tasks in comm, send the size of the map
///   to the recvtask and compute the total size on recvtask (llen)
///   map is a list of non-negative integers and the value of the integer indicates it's
///   position in the resulting array on recvtask, a value of 0 in map indicates a point
///   which is not sent.
///
int shr_spmd_get_gather_size(const MPI_Comm comm, const int maplen, MPI_Offset map[],
			     const int recvtask, int *scount,int *rcount, int *llen )
{
  int rank, ntasks;
  int i, j, rcnt;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &ntasks);

  *scount = 0;

  for(i=0;i < maplen; i++){
    if(map[i]>0){
      (*scount)++;
    }
  }

  if(rank==recvtask){
    rcnt = 1;
  }else{
    rcnt = 0;
  }

  shr_spmd_gather(&scount, 1, MPI_INT,
		rcount, rcnt, MPI_INT,
		recvtask, comm, maxreq);

  *llen = 0;

  if(rank==recvtask){
    for( i=0;i<ntasks;i++){
      (*llen) += rcount[i];
    }
  }
  return SHR_NOERR;
}

///
///  shr_create_comm_datatypes  Set up datatypes for a many to one or one to many communication
///
int shr_spmd_create_comm_datatypes(const MPI_Comm comm, const int maplen, MPI_Offset map[],
			      const int recvtask, MPI_Datatype basetype,
			      MPI_Datatype *sendtype, MPI_Datatype *recvtype)
{
  int rank, ntasks;
  int scount;
  int rcnt;
  MPI_Offset *sindex=NULL;
  int llen;
  mapsort *remap;
  int i, j;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &ntasks);
  int rcount[ntasks];

  shr_spmd_get_gather_size(comm, maplen,  map, recvtask, &scount, rcount, &llen );

  if(scount > 0){
    sindex = (MPI_Offset *) malloc(scount*sizeof(MPI_Offset));
    /* add error check */
    j=0;
    for(i=0;i<scount; i++){
      if(map[i]>0){
	sindex[j++]=i;
      }
    }
  }

  int rdispls[ntasks];
  int recvlths[ntasks];
  MPI_Offset *srcindex=NULL;
  if(rank==recvtask){
    rdispls[0] = 0;
    for( i=0;i<ntasks;i++){
      recvlths[i] = rcount[i];
      if(i>0){
	rdispls[i] = rdispls[i-1] + rcount[i-1];
      }
    }
    if(llen>0){
      srcindex = (MPI_Offset *)  malloc(llen*sizeof(MPI_Offset));
      /* add error check */
      for(i=0; i<llen; i++){
	srcindex[i]=0;
      }
    }
  }else{
    for(i=0;i<ntasks;i++){
      recvlths[i]=0;
      rdispls[i]=0;
    }
  }

  shr_spmd_gatherv(sindex, scount, MPI_OFFSET,
		 srcindex, recvlths, rdispls, MPI_OFFSET,
		 recvtask, comm, maxreq);


  MPI_Offset *shrtmap;
  MPI_Offset *gmap;
  gmap = (MPI_Offset *) malloc(llen*sizeof(MPI_Offset));

  if(maplen> scount && scount>0){
    shrtmap = (MPI_Offset *) malloc(scount*sizeof(MPI_Offset));
    /* add error check */

    j=0;
    for(i=0;i<maplen;i++)
      if(map[i]>0){
	shrtmap[j++]=map[i];
      }
  }else{
    shrtmap = map;
  }

  shr_spmd_gatherv((void *) shrtmap, scount, MPI_OFFSET,
		 (void *) gmap, recvlths, rdispls, MPI_OFFSET,
		 recvtask, comm, maxreq);


  if(shrtmap != map)
    free(shrtmap);
  int *rfrom=NULL;
  MPI_Offset *rindex=NULL;
  if(rank==recvtask && llen>0){
    int pos=0;
    int k=0;
    mapsort *mptr;
    for(i=0;i<ntasks;i++){
      for(j=0;j<rcount[i];j++){
	mptr = remap+k;
	mptr->rfrom = i;
	mptr->soffset = srcindex[pos+j];
	mptr->gmap = gmap[pos+j];
	k++;
      }
      pos += rcount[i];
    }
    // sort the mapping, this will transpose the data into IO order
    qsort(remap, llen, sizeof(mapsort), compare_offsets);

    rindex = (MPI_Offset *) malloc(llen*sizeof(MPI_Offset));
    /* add error check */
    rfrom = malloc(llen*sizeof(int));
    /* add error check */

    for(i=0;i<llen;i++){
      rindex[i]=0;
      rfrom[i]=0;
    }
  }
  int cnt[ntasks];
  int sndlths[ntasks];
  int sdispls[ntasks];
  MPI_Datatype dtypes[ntasks];
  for(i=0;i<ntasks;i++){
    cnt[i]=rdispls[i];

    /* offsets to swapm are in bytes */
    //    rdispls[i]*=pio_offset_size;
    sdispls[i]=0;
    sndlths[i]=0;
    dtypes[i]=MPI_OFFSET;
  }
  sndlths[0]=scount;
  mapsort *mptr;
  for(i=0;i<llen;i++){
    mptr = remap+i;
    rfrom[i] = mptr->rfrom;
    rindex[i]=i;
    gmap[i] = mptr->gmap;
    srcindex[ (cnt[rfrom[i]])++   ]=mptr->soffset;
  }

  shr_spmd_CheckMPIReturn(MPI_Scatterv((void *) srcindex, recvlths, rdispls, MPI_OFFSET,
			      (void *) sindex, scount,  MPI_OFFSET,
			      recvtask, comm),__FILE__,__LINE__);

  /*  still needs work
  if(rank == recvtask) {
    shr_create_mpi_datatype(basetype, ntasks, llen, rindex, rcount, rfrom, recvtype);
  }
  shr_create_mpi_datatype(basetype, 1, scount, sindex, scount, NULL, sendtype);
  */
}

int shr_spmd_create_comm_datatypes_finterface(const int comm, const int maplen, MPI_Offset map[],
			      const int recvtask, const int basetype,
			      int *sendtype, int *recvtype)
{
  MPI_Datatype csendtype, crecvtype;
  int ierr;

  ierr = shr_spmd_create_comm_datatypes(MPI_Comm_f2c(comm), maplen, map, recvtask,
					MPI_Type_f2c(basetype), &csendtype, &crecvtype);

  *sendtype = MPI_Type_c2f(csendtype);
  *recvtype = MPI_Type_c2f(crecvtype);

  return(ierr);

}



///
///  shr_spmd_gather provides the functionality of MPI_Gather with flow control options
///

int shr_spmd_gather( void *sendbuf, const int sendcnt, const MPI_Datatype sendtype,
		   void *recvbuf, const int recvcnt, const MPI_Datatype recvtype, const int root,
		   const MPI_Comm comm, const int flow_cntl)
{

  int gather_block_size;
  int mytask, nprocs;
  int mtag;
  MPI_Status status;
  int ierr;
  int hs;
  int displs;
  int dsize;

  if(flow_cntl > 0){

    gather_block_size = min(flow_cntl,MAX_GATHER_BLOCK_SIZE);

    shr_spmd_CheckMPIReturn(MPI_Comm_rank (comm, &mytask), __FILE__,__LINE__);
    shr_spmd_CheckMPIReturn(MPI_Comm_size (comm, &nprocs), __FILE__,__LINE__);

    mtag = 2*nprocs;
    hs = 1;

    if(mytask == root){
      int preposts = min(nprocs-1, gather_block_size);
      int head=0;
      int count=0;
      int tail = 0;
      int p;
      MPI_Request rcvid[gather_block_size];

      shr_spmd_CheckMPIReturn(MPI_Type_size(recvtype, &dsize), __FILE__,__LINE__);

      for( p=0;p<nprocs;p++){
	if(p != root){
	  if(recvcnt > 0){
	    count++;
	    if(count > preposts){
	      shr_spmd_CheckMPIReturn(MPI_Wait(rcvid+tail, &status), __FILE__,__LINE__);
	      tail = (tail+1) % preposts;
	    }
	    displs = p*recvcnt*dsize;

	    char *ptr = (char *) recvbuf + displs;

	    shr_spmd_CheckMPIReturn(MPI_Irecv(  ptr, recvcnt, recvtype, p, mtag, comm, rcvid+head), __FILE__,__LINE__);
	    head= (head+1) % preposts;
	    shr_spmd_CheckMPIReturn(MPI_Send( &hs, 1, MPI_INT, p, mtag, comm), __FILE__,__LINE__);
	  }
	}
      }

      // copy local data
      shr_spmd_CheckMPIReturn(MPI_Type_size(sendtype, &dsize), __FILE__,__LINE__);
      memcpy(recvbuf, sendbuf, sendcnt*dsize );

      count = min(count, preposts);

      if(count>0){
	shr_spmd_CheckMPIReturn(MPI_Waitall( count, rcvid, MPI_STATUSES_IGNORE),__FILE__,__LINE__);
      }
    }else{
      if(sendcnt > 0){
	shr_spmd_CheckMPIReturn(MPI_Recv( &hs, 1, MPI_INT, root, mtag, comm, &status), __FILE__,__LINE__);
	shr_spmd_CheckMPIReturn(MPI_Send( sendbuf, sendcnt, sendtype, root, mtag, comm), __FILE__,__LINE__);
      }
    }
  }else{
    shr_spmd_CheckMPIReturn(MPI_Gather ( sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm), __FILE__,__LINE__);
  }

  return SHR_NOERR;
}


///
///  shr_spmd_gather_finterface provides a fortran interface to shr_spmd_gather
///

int shr_spmd_gather_finterface( void *sendbuf, const int sendcnt, const int fsendtype,
		   void *recvbuf, const int recvcnt, const int frecvtype, const int root,
		   const int fcomm, int *flow_cntl)
{
  int flowcntl;
  MPI_Datatype sendtype, recvtype;
  int ierr;
  sendtype = MPI_Type_f2c(fsendtype);
  recvtype =MPI_Type_f2c(frecvtype);


  if(flow_cntl == NULL){
    flowcntl = 0;
  }else{
    flowcntl = *flow_cntl;
  }
  //  printf("line %d flowcntl=%d %x %x\n",__LINE__,flowcntl, sendbuf, recvbuf);
  ierr = shr_spmd_gather(sendbuf, sendcnt, sendtype,
			 recvbuf, recvcnt, recvtype, root, MPI_Comm_f2c(fcomm), flowcntl);

}




///
///  shr_spmd_gatherv provides the functionality of MPI_Gatherv with flow control options
///


int shr_spmd_gatherv(void *sendbuf, int sendcnt,  MPI_Datatype sendtype,
		    void *recvbuf,  int recvcnts[], int displs[],
		    MPI_Datatype recvtype, int root,
		    MPI_Comm comm, const int flow_cntl)
{
  bool fc_gather;
  int gather_block_size;
  int mytask, nprocs;
  int mtag;
  MPI_Status status;
  int ierr;
  int hs;
  int dsize;


  if(flow_cntl > 0){
    fc_gather = true;
    gather_block_size = min(flow_cntl,MAX_GATHER_BLOCK_SIZE);
  }else{
    fc_gather = false;
  }

  if(fc_gather){
    shr_spmd_CheckMPIReturn(MPI_Comm_rank (comm, &mytask), __FILE__,__LINE__);
    shr_spmd_CheckMPIReturn(MPI_Comm_size (comm, &nprocs), __FILE__,__LINE__);

    mtag = 2*nprocs;
    hs = 1;

    if(mytask == root){
      int preposts = min(nprocs-1, gather_block_size);
      int head=0;
      int count=0;
      int tail = 0;
      int p;
      MPI_Request rcvid[gather_block_size];
      // printf("%s %d %d\n",__FILE__,__LINE__,(int) recvtype);
      shr_spmd_CheckMPIReturn(MPI_Type_size(recvtype, &dsize), __FILE__,__LINE__);

      for( p=0;p<nprocs;p++){
	if(p != root){
	  if(recvcnts[p] > 0){
	    count++;
	    if(count > preposts){
	      shr_spmd_CheckMPIReturn(MPI_Wait(rcvid+tail, &status), __FILE__,__LINE__);
	      tail = (tail+1) % preposts;
	    }

	    void *ptr = (void *)((char *) recvbuf + dsize*displs[p]);

	    //	  printf("%s %d %d %d\n",__FILE__,__LINE__,p,(int) recvtype);
	    shr_spmd_CheckMPIReturn(MPI_Irecv(  ptr, recvcnts[p], recvtype, p, mtag, comm, rcvid+head), __FILE__,__LINE__);
	    head= (head+1) % preposts;
	    shr_spmd_CheckMPIReturn(MPI_Send( &hs, 1, MPI_INT, p, mtag, comm), __FILE__,__LINE__);
	  }
	}
      }

      // copy local data
      shr_spmd_CheckMPIReturn(MPI_Type_size(sendtype, &dsize), __FILE__,__LINE__);
      shr_spmd_CheckMPIReturn(MPI_Sendrecv(sendbuf, sendcnt, sendtype,
				  mytask, 102, recvbuf, recvcnts[mytask], recvtype,
				  mytask, 102, comm, &status),__FILE__,__LINE__);

      count = min(count, preposts);
      if(count>0)
	shr_spmd_CheckMPIReturn(MPI_Waitall( count, rcvid, MPI_STATUSES_IGNORE),__FILE__,__LINE__);
    }else{
      if(sendcnt > 0){
	shr_spmd_CheckMPIReturn(MPI_Recv( &hs, 1, MPI_INT, root, mtag, comm, &status), __FILE__,__LINE__);
	shr_spmd_CheckMPIReturn(MPI_Send( sendbuf, sendcnt, sendtype, root, mtag, comm), __FILE__,__LINE__);
      }
    }
  }else{
    shr_spmd_CheckMPIReturn(MPI_Gatherv ( sendbuf, sendcnt, sendtype, recvbuf, recvcnts, displs, recvtype, root, comm), __FILE__,__LINE__);
  }

  return SHR_NOERR;
}


///
///  shr_spmd_gather_finterface provides a fortran interface to shr_spmd_gather
///

int shr_spmd_gatherv_finterface(void *sendbuf, int sendcnt, const int fsendtype,
			       void *recvbuf, int recvcnts[],  int displs[],
			       const int frecvtype, int root, const int fcomm, const int flow_cntl)
{

  return(shr_spmd_gatherv(sendbuf, sendcnt, MPI_Type_f2c(fsendtype),
			recvbuf, recvcnts, displs, MPI_Type_f2c(frecvtype), root,
			MPI_Comm_f2c(fcomm), flow_cntl));

}





///
///  Returns the smallest power of 2 greater than i
///
int ceil2(const int i)
{
  int p=1;
  while(p<i){
    p*=2;
  }
  return(p);
}

///
///  Given integers p and k between 0 and np-1
///
int pair(const int np, const int p, const int k)
{
  int q = (p+1) ^ k ;
  int pair = (q > np-1)? -1: q;
  return pair;
}



/**
 * Provides the functionality of MPI_Alltoallw with flow control
 * options. Generalized all-to-all communication allowing different
 * datatypes, counts, and displacements for each partner
 *
 * @param sendbuf starting address of send buffer
 * @param sendcounts integer array equal to the number of tasks in
 * communicator comm (ntasks). It specifies the number of elements to
 * send to each processor
 * @param sdispls integer array (of length ntasks). Entry j
 * specifies the displacement in bytes (relative to sendbuf) from
 * which to take the outgoing data destined for process j.
 * @param sendtypes array of datatypes (of length ntasks). Entry j
 * specifies the type of data to send to process j.
 * @param recvbuf address of receive buffer.
 * @param recvcounts integer array (of length ntasks) specifying the
 * number of elements that can be received from each processor.
 * @param rdispls integer array (of length ntasks). Entry i
 * specifies the displacement in bytes (relative to recvbuf) at which
 * to place the incoming data from process i.
 * @param recvtypes array of datatypes (of length ntasks). Entry i
 * specifies the type of data received from process i.
 * @param comm MPI communicator for the MPI_Alltoallw call.
 * @param handshake if true, use handshaking.
 * @param isend the isend bool indicates whether sends should be
 * posted using mpi_irsend which can be faster than blocking
 * sends. When flow control is used max_requests > 0 and the number of
 * irecvs posted from a given task will not exceed this value. On some
 * networks too many outstanding irecvs will cause a communications
 * bottleneck.
 * @param max_requests If 0, no flow control is used.
 * @returns 0 for success, error code otherwise.
 */
int shr_spmd_swapm(void *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype *sendtypes,
	      void *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype *recvtypes,
	      MPI_Comm comm, const bool handshake, bool isend, const int max_requests)
{
    int ntasks;  /* Number of tasks in communicator comm. */
    int my_rank; /* Rank of this task in comm. */
    int maxsend = 0;
    int maxrecv = 0;
    int tag;
    int offset_t;
    int steps;
    int istep;
    int rstep;
    int p;
    int maxreq;
    int maxreqh;
    int hs = 1; /* Used for handshaking. */
    int cnt;
    void *ptr;
    MPI_Status status; /* Not actually used - replace with MPI_STATUSES_IGNORE. */
    int mpierr;  /* Return code from MPI functions. */
    int ierr;    /* Return value. */


    /* Get my rank and size of communicator. */
    if ((mpierr = MPI_Comm_size(comm, &ntasks)))
	return shr_spmd_CheckMPIReturn( mpierr, __FILE__, __LINE__);
    if ((mpierr = MPI_Comm_rank(comm, &my_rank)))
	return shr_spmd_CheckMPIReturn( mpierr, __FILE__, __LINE__);

    /* Now we know the size of these arrays. */
    int swapids[ntasks];
    MPI_Request rcvids[ntasks];
    MPI_Request sndids[ntasks];
    MPI_Request hs_rcvids[ntasks];

    /* If max_requests == 0 no throttling is requested and the default
     * mpi_alltoallw function is used. */
    if (max_requests == 0)
    {
#ifdef OPEN_MPI
	/* OPEN_MPI developers determined that MPI_DATATYPE_NULL was
	   not a valid argument to MPI_Alltoallw according to the
	   standard. The standard is a little vague on this issue and
	   other mpi vendors disagree. In my opinion it just makes
	   sense that if an argument expects an mpi datatype then
	   MPI_DATATYPE_NULL should be valid. For each task it's
	   possible that either sendlength or receive length is 0 it
	   is in this case that the datatype null makes sense. */
	for (int i = 0; i < ntasks; i++)
	{
	    if (sendtypes[i] == MPI_DATATYPE_NULL)
		sendtypes[i] = MPI_CHAR;
	    if (recvtypes[i] == MPI_DATATYPE_NULL)
		recvtypes[i] = MPI_CHAR;
	}
#endif

	/* Call the MPI alltoall without flow control. */
	if ((mpierr = MPI_Alltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf,
				    recvcounts, rdispls, recvtypes, comm)))
	    return shr_spmd_CheckMPIReturn( mpierr, __FILE__, __LINE__);

#ifdef OPEN_MPI
	/* OPEN_MPI has problems with MPI_DATATYPE_NULL. */
	for (int i = 0; i < ntasks; i++)
	{
	    if (sendtypes[i] == MPI_CHAR)
		sendtypes[i] = MPI_DATATYPE_NULL;
	    if (recvtypes[i] == MPI_CHAR)
		recvtypes[i] = MPI_DATATYPE_NULL;
	}
#endif
	return SHR_NOERR;
    }

    /* an index for communications tags */
    offset_t = ntasks;

    /* Send to self. */
    if (sendcounts[my_rank] > 0)
    {
	void *sptr, *rptr;
	int extent, lb;
	tag = my_rank + offset_t;
	sptr = (char *)sendbuf + sdispls[my_rank];
	rptr = (char *)recvbuf + rdispls[my_rank];

	/*
	  MPI_Type_get_extent(sendtypes[my_rank], &lb, &extent);
	  printf("%s %d %d %d\n",__FILE__,__LINE__,extent, lb);
	  MPI_Type_get_extent(recvtypes[my_rank], &lb, &extent);
	  printf("%s %d %d %d\n",__FILE__,__LINE__,extent, lb);
	*/

#ifdef ONEWAY
	/* If ONEWAY is true we will post mpi_sendrecv comms instead
	 * of irecv/send. */
	if ((mpierr = MPI_Sendrecv(sptr, sendcounts[my_rank],sendtypes[my_rank],
				   my_rank, tag, rptr, recvcounts[my_rank], recvtypes[my_rank],
				   my_rank, tag, comm, &status)))
	    return shr_spmd_CheckMPIReturn( mpierr, __FILE__, __LINE__);
#else
	if ((mpierr = MPI_Irecv(rptr, recvcounts[my_rank], recvtypes[my_rank],
				my_rank, tag, comm, rcvids)))
	    return shr_spmd_CheckMPIReturn( mpierr, __FILE__, __LINE__);
	if ((mpierr = MPI_Send(sptr, sendcounts[my_rank], sendtypes[my_rank],
			       my_rank, tag, comm)))
	    return shr_spmd_CheckMPIReturn( mpierr, __FILE__, __LINE__);

	if ((mpierr = MPI_Wait(rcvids, &status)))
	    return shr_spmd_CheckMPIReturn( mpierr, __FILE__, __LINE__);
#endif
    }

    /* When send to self is complete there is nothing left to do if
     * ntasks==1. */
    if (ntasks == 1)
	return SHR_NOERR;

    for (int i = 0; i < ntasks; i++)
    {
	rcvids[i] = MPI_REQUEST_NULL;
	swapids[i] = 0;
    }

    if (isend)
	for (int i = 0; i < ntasks; i++)
	    sndids[i] = MPI_REQUEST_NULL;

    if (handshake)
	for (int i = 0; i < ntasks; i++)
	    hs_rcvids[i] = MPI_REQUEST_NULL;

    steps = 0;
    for (istep = 0; istep < ceil2(ntasks) - 1; istep++)
    {
	p = pair(ntasks, istep, my_rank);
	if (p >= 0 && (sendcounts[p] > 0 || recvcounts[p] > 0))
	    swapids[steps++] = p;
    }

    if (steps == 1)
    {
	maxreq = 1;
	maxreqh = 1;
    }
    else
    {
	if (max_requests > 1 && max_requests < steps)
	{
	    maxreq = max_requests;
	    maxreqh = maxreq / 2;
	}
	else if (max_requests >= steps)
	{
	    maxreq = steps;
	    maxreqh = steps;
	}
	else
	{
	    maxreq = 2;
	    maxreqh = 1;
	}
    }

    /* If handshaking is in use, do a nonblocking recieve to listen
     * for it. */
    if (handshake)
    {
	for (istep = 0; istep < maxreq; istep++)
	{
	    p = swapids[istep];
	    if (sendcounts[p] > 0)
	    {
		tag = my_rank + offset_t;
		if ((mpierr = MPI_Irecv(&hs, 1, MPI_INT, p, tag, comm, hs_rcvids + istep)))
		    return shr_spmd_CheckMPIReturn( mpierr, __FILE__, __LINE__);
	    }
	}
    }

    /* Post up to maxreq irecv's. */
    printf ("post irecv maxreq = %d\n", maxreq);
    for (istep = 0; istep < maxreq; istep++)
    {
	p = swapids[istep];
	if (recvcounts[p] > 0)
	{
	    tag = p + offset_t;
	    ptr = (char *)recvbuf + rdispls[p];

	    if ((mpierr = MPI_Irecv(ptr, recvcounts[p], recvtypes[p], p, tag, comm,
				    rcvids + istep)))
		return shr_spmd_CheckMPIReturn( mpierr, __FILE__, __LINE__);
	    printf("irecs rcvids[%d] = %d tag=%d\n",istep,rcvids[istep], tag);

	    if (handshake)
		if ((mpierr = MPI_Send(&hs, 1, MPI_INT, p, tag, comm)))
		    return shr_spmd_CheckMPIReturn( mpierr, __FILE__, __LINE__);
	}
    }

    /* Tell the paired task that this tasks' has posted it's irecvs'. */
    rstep = maxreq;
    for (istep = 0; istep < steps; istep++)
    {
	p = swapids[istep];
	if (sendcounts[p] > 0)
	{
	    tag = my_rank + offset_t;
	    /* If handshake is enabled don't post sends until the
	     * receiving task has posted recvs. */
	    if (handshake)
	    {
		if ((mpierr = MPI_Wait(hs_rcvids + istep, &status)))
		    return shr_spmd_CheckMPIReturn( mpierr, __FILE__, __LINE__);
		hs_rcvids[istep] = MPI_REQUEST_NULL;
	    }
	    ptr = (char *)sendbuf + sdispls[p];

	    if (isend)
		if ((mpierr = MPI_Irsend(ptr, sendcounts[p], sendtypes[p], p, tag, comm,
					 sndids + istep)))
		    return shr_spmd_CheckMPIReturn( mpierr, __FILE__, __LINE__);
	    else
		if ((mpierr = MPI_Send(ptr, sendcounts[p], sendtypes[p], p, tag, comm)))
		    return shr_spmd_CheckMPIReturn( mpierr, __FILE__, __LINE__);
	    printf("send tag= %d\n",tag);
	}

	printf("istep %d maxreqh %d\n",istep, maxreqh);
	/* We did comms in sets of size max_reqs, if istep > maxreqh
	 * then there is a remainder that must be handled. */
	if (istep > maxreqh)
	{
	    p = istep - maxreqh;
	    if (rcvids[p] != MPI_REQUEST_NULL)
	    {
		if ((mpierr = MPI_Wait(rcvids + p, &status)))
		    return shr_spmd_CheckMPIReturn( mpierr, __FILE__, __LINE__);
		rcvids[p] = MPI_REQUEST_NULL;
	    }
	    if (rstep < steps)
	    {
		p = swapids[rstep];
		if (handshake && sendcounts[p] > 0)
		{
		    tag = my_rank + offset_t;
		    if ((mpierr = MPI_Irecv(&hs, 1, MPI_INT, p, tag, comm, hs_rcvids+rstep)))
			return shr_spmd_CheckMPIReturn( mpierr, __FILE__, __LINE__);
		}
		if (recvcounts[p] > 0)
		{
		    tag = p + offset_t;

		    ptr = (char *)recvbuf + rdispls[p];
		    if ((mpierr = MPI_Irecv(ptr, recvcounts[p], recvtypes[p], p, tag, comm, rcvids + rstep)))
			return shr_spmd_CheckMPIReturn( mpierr, __FILE__, __LINE__);
		    if (handshake)
			if ((mpierr = MPI_Send(&hs, 1, MPI_INT, p, tag, comm)))
			    return shr_spmd_CheckMPIReturn( mpierr, __FILE__, __LINE__);
		}
		rstep++;
	    }
	}
    }
    /* If steps > 0 there are still outstanding messages, wait for
     * them here. */
    if (steps > 0)
    {
	for (int i=0; i< steps; i++)
	    printf("rcvids[%d] = %d\n", i, rcvids[i]);
	printf("\n");

	if ((mpierr = MPI_Waitall(steps, rcvids, MPI_STATUSES_IGNORE)))
	    return shr_spmd_CheckMPIReturn( mpierr, __FILE__, __LINE__);

	if (isend)
	    if ((mpierr = MPI_Waitall(steps, sndids, MPI_STATUSES_IGNORE)))
		return shr_spmd_CheckMPIReturn( mpierr, __FILE__, __LINE__);
    }
    printf("steps %d\n",steps);

    return SHR_NOERR;
}



///
///  shr_spmd_gather_finterface provides a fortran interface to shr_spmd_gather
///

int shr_spmd_swapm_finterface( void *sendbuf, int sendcnts[],  int sdispls[], int fsendtypes[],
			  void *recvbuf, int recvcnts[],  int rdispls[],
			  int frecvtypes[], const int fcomm, const int handshake,
			  const int isend, const int max_reqs)
{
  int npes;
  MPI_Datatype *sendtypes, *recvtypes;
  MPI_Comm comm;
  int ierr;
  int i;

  comm = MPI_Comm_f2c(fcomm);
  ierr = MPI_Comm_size(comm, &npes);
  sendtypes = (MPI_Datatype *) malloc(npes*sizeof(MPI_Datatype));
  recvtypes = (MPI_Datatype *) malloc(npes*sizeof(MPI_Datatype));

  for(i=0;i<npes; i++){
    sendtypes[i]=MPI_Type_f2c(fsendtypes[i]);
    recvtypes[i]=MPI_Type_f2c(frecvtypes[i]);
  }

  ierr = shr_spmd_swapm(sendbuf, sendcnts, sdispls, sendtypes,
		   recvbuf, recvcnts, rdispls, recvtypes,
			comm, (handshake!=0), (isend!=0), max_reqs);
  free(sendtypes);
  free(recvtypes);
  return(ierr);
}
