import math
from algorithm import vectorize_unroll, vectorize

alias simd_width: Int = simdwidthof[DType.int8]()
alias T = DType.int8

@always_inline
fn arg_true[simd_width: Int](v: SIMD[DType.bool, simd_width]) -> Int:
    for i in range(simd_width):
        if v[i]:
            return i
    return -1

@always_inline
fn find_chr_next_occurance[T: DType](in_tensor: Tensor[T], chr: Int, start: Int = 0) -> Int:
    let len = in_tensor.num_elements() - start
    for s in range(start, start+len, simd_width):
        let v = in_tensor.simd_load[simd_width](s)
        # print("v=",v)
        let mask = v == chr
        if mask.reduce_or():
            return s + arg_true(mask)

    return -1

@always_inline
fn slice_tensor[T: DType](in_tensor: Tensor[T], start: Int, end: Int) -> Tensor[T]:
    if start >= end:
        return Tensor[T](0)

    var out_tensor: Tensor[T] = Tensor[T](end - start)

    @parameter
    fn inner[simd_width: Int](size: Int):
        let transfer = in_tensor.simd_load[simd_width](start + size)
        out_tensor.simd_store[simd_width](size, transfer)

    vectorize[simd_width, inner](out_tensor.num_elements())

    return out_tensor

@always_inline
fn get_next_segment[T: DType](in_tensor: Tensor[T], start: Int, chr: Int = 9) -> Tensor[T]:
    var in_start = start
    while in_tensor[in_start] == chr:
        in_start += 1
        if in_start >= in_tensor.num_elements():
            return Tensor[T](0)

    var next_pos = find_chr_next_occurance(in_tensor, chr, in_start)
    if next_pos == -1:
        next_pos = in_tensor.num_elements()
    return slice_tensor(in_tensor, in_start, next_pos)

@always_inline
fn read_bytes(
    handle: FileHandle, beginning: UInt64, length: Int64
) raises -> Tensor[DType.int8]:
    """Function to read a file chunk given an offset to seek.
    Returns empty tensor or under-sized tensor if reached EOF.
    """
    _ = handle.seek(beginning)
    return handle.read_bytes(length)

@always_inline
fn write_to_buff[T: DType](src: Tensor[T], inout dest: Tensor[T], start: Int):
    """
    Copy a small tensor into a larger tensor given an index at the large tensor.
    Implemented iteratively due to small gain from copying less then 1MB tensor using SIMD.
    Assumes copying is always in bounds. Bound checking is th responsbility of the caller.
    """
    for i in range(src.num_elements()):
        dest[start + i] = src[i]

    # End of helper function
    #==========================================================================

@value
struct ReadSegment:
    var QName: Tensor[T]           # Query name template
    var FLAG: Tensor[T]               # Combination of bitwise FLAGs.(https://en.wikipedia.org/wiki/Bit_field)
    var CHR: Tensor[T]           # Reference Chromosome
    var POS: Tensor[T]                # 1-based position for SAM (0-based for BAM)
    var MapQ: Tensor[T]               # -10log10Pr{mapping position is wrong}
    var CIGAR: Tensor[T]           # Match-mismatch string  (81M2D37M27S)
    var RNEXT: Tensor[T]           # reference sequence name, = means identical to RName
    var PNEXT: Tensor[T]              # Position of of next read
    var TLEN: Tensor[T]               # Template LENgth 
    var Seq: Tensor[T]             # segment SEQuence
    var Qual: Tensor[T]            # ASCII of base QUALity plus 33 (phred-score)

    @always_inline
    fn get_Segment_Name(self) -> String:
        var t = self.QName
        return String(t._steal_ptr())

    @always_inline
    fn get_MapQ(self) -> String:
        var t = self.MapQ
        return String(t._steal_ptr())
    
    @always_inline
    fn get_seq(self) -> String:
        var t = self.Seq
        return String(t._steal_ptr())

    fn get_chr(self) -> String:
        var t = self.CHR
        return String(t._steal_ptr())

    fn get_pos(self) -> String:
        var t = self.POS
        return String(t._steal_ptr())

    @always_inline
    fn _concatenate(self) -> Tensor[T]:
        var offset = 0
        let total_length = self.QName.num_elements() + 1 + self.FLAG.num_elements() + 1 + self.CHR.num_elements() + 1 + self.POS.num_elements() + 1 + self.MapQ.num_elements() + 1 + self.CIGAR.num_elements() + 1 + self.RNEXT.num_elements() + 1 + self.PNEXT.num_elements() + 1 + self.TLEN.num_elements() + 1 + self.Seq.num_elements() + 1 + self.Qual.num_elements() + 1
        var out_tensor = Tensor[T](total_length)

        write_to_buff(self.QName, out_tensor, offset)
        offset += self.QName.num_elements() + 1
        out_tensor[offset - 1] = 9

        write_to_buff(self.FLAG, out_tensor, offset)
        offset += self.FLAG.num_elements() + 1
        out_tensor[offset - 1] = 9

        write_to_buff(self.CHR, out_tensor, offset)
        offset += self.CHR.num_elements() + 1
        out_tensor[offset - 1] = 9

        write_to_buff(self.POS, out_tensor, offset)
        offset += self.POS.num_elements() + 1
        out_tensor[offset - 1] = 9

        write_to_buff(self.MapQ, out_tensor, offset)
        offset += self.MapQ.num_elements() + 1
        out_tensor[offset - 1] = 9

        write_to_buff(self.CIGAR, out_tensor, offset)
        offset += self.CIGAR.num_elements() + 1
        out_tensor[offset - 1] = 9

        write_to_buff(self.RNEXT, out_tensor, offset)
        offset += self.RNEXT.num_elements() + 1
        out_tensor[offset - 1] = 9

        write_to_buff(self.PNEXT, out_tensor, offset)
        offset += self.PNEXT.num_elements() + 1
        out_tensor[offset - 1] = 9

        write_to_buff(self.TLEN, out_tensor, offset)
        offset += self.TLEN.num_elements() + 1
        out_tensor[offset - 1] = 9

        write_to_buff(self.Seq, out_tensor, offset)
        offset += self.Seq.num_elements() + 1
        out_tensor[offset - 1] = 9

        write_to_buff(self.Qual, out_tensor, offset)
        offset += self.Qual.num_elements() + 1
        out_tensor[offset - 1] = 9

        return out_tensor

    @always_inline
    fn __str__(self) -> String:
        var t = self._concatenate()
        return "ReadSegment\t" + String(t._steal_ptr())

    @always_inline
    fn __repr__(self) -> String:
        return self.__str__()

struct SamParser:
    var _file_handle: FileHandle
    var _BUF_SIZE: Int
    var _current_chunk: Tensor[T]
    var _file_pos: Int
    var _chunk_last_index: Int
    var _chunk_pos: Int

    fn __init__(inout self, path: String, BUF_SIZE: Int = 64 * 1024) raises -> None:
        self._BUF_SIZE = BUF_SIZE
        self._file_pos = 0
        self._current_chunk = Tensor[T](0)
        self._chunk_last_index = 0
        self._chunk_pos = 0
        
        self._file_handle = open(path, "r")
        self.fill_buffer()
    
    @always_inline
    fn fill_buffer(inout self) raises:
        self._current_chunk = read_bytes(self._file_handle, self._file_pos, self._BUF_SIZE)
    
    @always_inline
    fn check_EOF(self) raises:
        if self._current_chunk.num_elements() == 0:
            raise Error("EOF")

    @always_inline
    fn next(inout self) raises -> ReadSegment:
        let read: ReadSegment

        self.check_EOF()
        if self._chunk_pos >= self._chunk_last_index:
            self.fill_buffer()

        read = self.parse_read(self._chunk_pos, self._current_chunk)

        return read

    @always_inline
    fn parse_read(self, inout pos: Int, chunk: Tensor[T]) raises -> ReadSegment:
        let read: ReadSegment
  
        let QN = get_next_segment(chunk, pos)
        pos += QN.num_elements() + 1
        let Flag = get_next_segment(chunk, pos)
        pos += Flag.num_elements() + 1
        let rname = get_next_segment(chunk, pos)
        pos += rname.num_elements() + 1
        let rpos = get_next_segment(chunk, pos)
        pos += rpos.num_elements() + 1
        let mapq =  get_next_segment(chunk, pos)
        pos += mapq.num_elements() + 1
        let cigar =  get_next_segment(chunk, pos)
        pos += cigar.num_elements() + 1
        let rnext = get_next_segment(chunk, pos)
        pos += rnext.num_elements() + 1
        let pnext = get_next_segment(chunk, pos)
        pos += pnext.num_elements() + 1
        let tlen = get_next_segment(chunk, pos)
        pos += tlen.num_elements() + 1
        let seq = get_next_segment(chunk, pos)
        pos += seq.num_elements() + 1
        let qual = get_next_segment(chunk, pos)
        pos += qual.num_elements() + 1
        
        read = ReadSegment(QN, Flag, rname, rpos, mapq, cigar, rnext, pnext, tlen, seq, qual)
        
        pos = find_chr_next_occurance(chunk, 10, pos) + 1

        return read

    @always_inline
    fn print_buffer(inout self) raises:
        print(self._current_chunk)


fn main() raises:
    let bamFile = "/mnt/d/JorritvU/Tripolar/scDNA-seq/CHI-005/processed/CHI-005_bi_99_Bipolar.sam"
    var sam: SamParser
    sam = SamParser(bamFile)
    

    var read = sam.next()
    var i = 0
    while read.QName.num_elements() > 0 and i < 250:
        print(read.__str__() + "\n")

        var position = read.get_chr() + ":" + read.get_pos()   
        print(position + "\n")
        read = sam.next()
        i += 1

    sam._file_handle.close()
