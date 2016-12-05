#ifndef __SHR_SPMD__
#define __SHR_SPMD__
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h> // memcpy

#define MAX_GATHER_BLOCK_SIZE 8192
#ifndef min
#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })
#endif

#define SHR_NOERR 0

#if defined(__cplusplus)
extern "C" {
#endif
  int shr_spmd_CheckMPIReturn(const int ierr,const char file[],const int line);
  int shr_spmd_get_gather_size(const MPI_Comm comm, const int maplen, MPI_Offset map[],
			       const int recvtask, int *scount,int *rcount, int *llen );
  int shr_spmd_gather( void *sendbuf, const int sendcnt, const MPI_Datatype sendtype,
			  void *recvbuf, const int recvcnt, const MPI_Datatype recvtype, const int root,
			  const MPI_Comm comm, const int flow_cntl);
  int shr_spmd_gather_finterface( void *sendbuf, const int sendcnt, const int fsendtype,
				     void *recvbuf,  const int recvcnt, const int frecvtype, const int root,
				     const int fcomm, int *flow_cntl);

int shr_spmd_gatherv(void *sendbuf, int sendcnt, MPI_Datatype sendtype,
		   void *recvbuf, int *recvcnts, int *displs,
		     MPI_Datatype recvtype, int root, MPI_Comm comm, int flow_cntl);

  int shr_spmd_gatherv_finterface( void *sendbuf, int sendcnt, const int fsendtype,
				  void *recvbuf,  int recvcnts[],  int displs[],
				  const int frecvtype, int root, const int fcomm, const int flow_cntl);

int shr_spmd_swapm(void *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype *sendtypes,
	      void *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype *recvtypes,
		   MPI_Comm comm, const bool handshake, bool isend, const int max_requests);


  int shr_spmd_swapm_finterface( void *sendbuf, int sendcnts[],  int sdispls[], int fsendtypes[],
				 void *recvbuf, int recvcnts[],  int rdispls[],
				 int frecvtypes[], const int fcomm, const int handshake,
				 const int isend, const int max_reqs);



#if defined(__cplusplus)
}
#endif
#endif
