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
void shr_spmd_CheckMPIReturn(const int ierr,const char file[],const int line)
{

  if(ierr != MPI_SUCCESS){
    char errstring[MPI_MAX_ERROR_STRING];
    int errstrlen;
    int mpierr = MPI_Error_string( ierr, errstring, &errstrlen);

    fprintf(stderr, "MPI ERROR: %s in file %s at line %d\n",errstring, file, line);

  }
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


///
///  shr_spmd_swapm provides the functionality of MPI_Alltoallw with flow control options
///

int shr_spmd_swapm(void *sndbuf,   int sndlths[], int sdispls[],  MPI_Datatype stypes[],
	      void *rcvbuf,  int rcvlths[],  int rdispls[],  MPI_Datatype rtypes[],
	       MPI_Comm comm,const  bool handshake, bool isend,const  int max_requests)
{

  int nprocs;
  int mytask;

  int maxsend=0;
  int maxrecv=0;
  int i;
  shr_spmd_CheckMPIReturn(MPI_Comm_size(comm, &nprocs),__FILE__,__LINE__);
  shr_spmd_CheckMPIReturn(MPI_Comm_rank(comm, &mytask),__FILE__,__LINE__);

  if(max_requests == 0) {
#ifdef DEBUG
    int totalrecv=0;
    int totalsend=0;
    for( i=0;i<nprocs;i++){
      //      printf("%d sndlths %d %d %d %d\n",i,sndlths[i],sdispls[i],rcvlths[i],rdispls[i]);
      totalsend+=sndlths[i];
      totalrecv+=rcvlths[i];
    }
    printf("%s %d totalsend %d totalrecv %d \n",__FILE__,__LINE__,totalsend,totalrecv);

#endif
    shr_spmd_CheckMPIReturn(MPI_Alltoallw( sndbuf, sndlths, sdispls, stypes, rcvbuf, rcvlths, rdispls, rtypes, comm),__FILE__,__LINE__);
    return SHR_NOERR;
  }

  int tag;
  int offset_t;
  int ierr;
  MPI_Status status;
  int steps;
  int istep;
  int rstep;
  int p;
  int maxreq;
  int maxreqh;
  int hs;
  int cnt;
  void *ptr;
  MPI_Request rcvids[nprocs];



  offset_t = nprocs;
  // send to self
  if(sndlths[mytask] > 0){
    void *sptr, *rptr;
    int extent, lb;
    tag = mytask + offset_t;
    sptr = (void *)((char *) sndbuf + sdispls[mytask]);
    rptr = (void *)((char *) rcvbuf  + rdispls[mytask]);

    /*
      MPI_Type_get_extent(stypes[mytask], &lb, &extent);
      printf("%s %d %d %d\n",__FILE__,__LINE__,extent, lb);
      MPI_Type_get_extent(rtypes[mytask], &lb, &extent);
      printf("%s %d %d %d\n",__FILE__,__LINE__,extent, lb);
    */
#ifdef ONEWAY
    shr_spmd_CheckMPIReturn(MPI_Sendrecv(sptr, sndlths[mytask],stypes[mytask],
				mytask, tag, rptr, rcvlths[mytask], rtypes[mytask],
				mytask, tag, comm, &status),__FILE__,__LINE__);
#else
    //   printf("%s %d \n",__FILE__,__LINE__);
    shr_spmd_CheckMPIReturn(MPI_Irecv(rptr, rcvlths[mytask], rtypes[mytask],
			     mytask, tag, comm, rcvids),__FILE__,__LINE__);
    //printf("%s %d \n",__FILE__,__LINE__);
    shr_spmd_CheckMPIReturn(MPI_Send(sptr, sndlths[mytask], stypes[mytask],
			     mytask, tag, comm),__FILE__,__LINE__);

    //printf("%s %d %d\n",__FILE__,__LINE__,rcvids[0]);
    shr_spmd_CheckMPIReturn(MPI_Wait(rcvids, &status),__FILE__,__LINE__);

#endif


  }
  if(nprocs==1)
    return SHR_NOERR;

  int swapids[nprocs];
  MPI_Request sndids[nprocs];
  MPI_Request hs_rcvids[nprocs];
  for( i=0;i<nprocs;i++){
    rcvids[i] = MPI_REQUEST_NULL;
    swapids[i]=0;
  }
  if(isend)
    for( i=0;i<nprocs;i++)
      sndids[i]=MPI_REQUEST_NULL;
  if(handshake)
    for( i=0;i<nprocs;i++)
      hs_rcvids[i]=MPI_REQUEST_NULL;

  steps = 0;
  for(istep=0;istep<ceil2(nprocs)-1;istep++){
    p = pair(nprocs, istep, mytask) ;
    if( p >= 0 && (sndlths[p] > 0 || rcvlths[p] > 0)){
      swapids[steps++] = p;
    }
  }

  if(steps == 1){
    maxreq = 1;
    maxreqh = 1;
  }else{
    if(max_requests > 1 && max_requests<steps){
      maxreq = max_requests;
      maxreqh = maxreq/2;
    }else if(max_requests>=steps){
      maxreq = steps;
      maxreqh = steps;
    }else{
      maxreq = 2;
      maxreqh = 1;
    }
  }
  if(handshake){
    hs = 1;
    for(istep=0; istep<maxreq; istep++){
      p = swapids[istep];
      if( sndlths[p] > 0){
	tag = mytask+offset_t;
	shr_spmd_CheckMPIReturn(MPI_Irecv( &hs, 1, MPI_INT, p, tag, comm, hs_rcvids+istep), __FILE__,__LINE__);
      }
    }
  }
  for(istep=0;istep < maxreq; istep++){
    p = swapids[istep];
    if(rcvlths[p] > 0){
      tag = p + offset_t;
      ptr = (void *)((char *) rcvbuf + rdispls[p]);

      //	  printf("%s %d %d %d\n",__FILE__,__LINE__,p,(int) rtypes[p]);
      shr_spmd_CheckMPIReturn(MPI_Irecv( ptr, rcvlths[p], rtypes[p], p, tag, comm, rcvids+istep), __FILE__,__LINE__);

      if(handshake)
	shr_spmd_CheckMPIReturn(MPI_Send( &hs, 1, MPI_INT, p, tag, comm), __FILE__,__LINE__);
    }
  }

  rstep = maxreq;
  for(istep = 0; istep < steps; istep++){
    p = swapids[istep];
    if(sndlths[p] > 0){
      tag = mytask + offset_t;
      if(handshake){
	shr_spmd_CheckMPIReturn(MPI_Wait ( hs_rcvids+istep, &status), __FILE__,__LINE__);
	hs_rcvids[istep] = MPI_REQUEST_NULL;
      }
      ptr = (void *)((char *) sndbuf + sdispls[p]);

      if(isend){
	shr_spmd_CheckMPIReturn(MPI_Irsend(ptr, sndlths[p], stypes[p], p, tag, comm,sndids+istep), __FILE__,__LINE__);
      }else{
	shr_spmd_CheckMPIReturn(MPI_Send(ptr, sndlths[p], stypes[p], p, tag, comm), __FILE__,__LINE__);
      }

    }
    if(istep > maxreqh){
      p = istep - maxreqh;
      if(rcvids[p] != MPI_REQUEST_NULL){
	shr_spmd_CheckMPIReturn(MPI_Wait(rcvids+p, &status), __FILE__,__LINE__);
	rcvids[p] = MPI_REQUEST_NULL;
      }
      if(rstep < steps){
	p = swapids[rstep];
	if(handshake && sndlths[p] > 0){
	  tag = mytask + offset_t;
	  shr_spmd_CheckMPIReturn(MPI_Irecv( &hs, 1, MPI_INT, p, tag, comm, hs_rcvids+rstep), __FILE__,__LINE__);
	}
	if(rcvlths[p] > 0){
	  tag = p + offset_t;

	  ptr = (void *)((char *) rcvbuf + rdispls[p]);
	  shr_spmd_CheckMPIReturn(MPI_Irecv( ptr, rcvlths[p], rtypes[p], p, tag, comm, rcvids+rstep), __FILE__,__LINE__);
	  if(handshake)
	    shr_spmd_CheckMPIReturn(MPI_Send( &hs, 1, MPI_INT, p, tag, comm), __FILE__,__LINE__);
	}
	rstep++;
      }
    }
  }
  //     printf("%s %d %d \n",__FILE__,__LINE__,nprocs);
  if(steps>0){
      shr_spmd_CheckMPIReturn(MPI_Waitall(steps, rcvids, MPI_STATUSES_IGNORE), __FILE__,__LINE__);
    if(isend)
      shr_spmd_CheckMPIReturn(MPI_Waitall(steps, sndids, MPI_STATUSES_IGNORE), __FILE__,__LINE__);
  }
  //      printf("%s %d %d \n",__FILE__,__LINE__,nprocs);

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
