// Syntax-only stub of the tt-metal host API used by render/host (task #99).
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#ifndef TT_FATAL
#define TT_FATAL(cond, ...) ((void)(cond))
#endif
namespace tt {
enum class DataFormat { Float32, UInt32, Float16_b, Int32, UInt8 };
namespace tt_metal {
using KernelHandle = uint32_t;
struct CoreCoord { std::size_t x = 0, y = 0; CoreCoord() = default; CoreCoord(std::size_t a, std::size_t b) : x(a), y(b) {} };
struct CoreRange { CoreCoord start_coord, end_coord; CoreRange(CoreCoord a, CoreCoord b) : start_coord(a), end_coord(b) {} };
struct CoreRangeSet { CoreRangeSet() = default; CoreRangeSet(CoreRange) {} CoreRangeSet(CoreCoord) {}
    std::size_t num_cores() const; const std::vector<CoreRange>& ranges() const; };
enum class MathFidelity { LoFi, HiFi2, HiFi3, HiFi4 };
enum class UnpackToDestMode { Default, UnpackToDestFp32 };
enum class DataMovementProcessor { RISCV_0, RISCV_1 };
enum class NOC { RISCV_0_default, RISCV_1_default, NOC_0, NOC_1 };
enum class BufferType { DRAM, L1 };
struct DataMovementConfig {
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0;
    NOC noc = NOC::RISCV_0_default;
    std::vector<uint32_t> compile_args;
    std::map<std::string, std::string> defines;
};
struct ComputeConfig {
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    bool dst_full_sync_en = false;
    std::vector<UnpackToDestMode> unpack_to_dest_mode;
    bool bfp8_pack_precise = false;
    bool math_approx_mode = false;
    std::vector<uint32_t> compile_args;
    std::map<std::string, std::string> defines;
};
struct CircularBufferConfig {
    CircularBufferConfig(uint32_t, const std::map<uint8_t, tt::DataFormat>&) {}
    CircularBufferConfig& set_page_size(uint32_t, uint32_t) { return *this; }
};
struct Program {};
Program CreateProgram();
KernelHandle CreateKernel(Program&, const std::string&, const CoreRangeSet&,
                          const std::variant<DataMovementConfig, ComputeConfig>&);
uint32_t CreateCircularBuffer(Program&, const CoreRangeSet&, const CircularBufferConfig&);
void SetRuntimeArgs(Program&, KernelHandle, const CoreCoord&, const std::vector<uint32_t>&);
uint32_t CreateSemaphore(Program&, const CoreRangeSet&, uint32_t);
namespace distributed { class MeshDevice; }
void ReadMeshDeviceProfilerResults(distributed::MeshDevice&);
struct TensorAccessorArgs {
    static TensorAccessorArgs create_dram_interleaved();
    void append_to(std::vector<uint32_t>&) const;
};
class Allocator { public: uint32_t get_num_banks(const BufferType&) const; };
namespace distributed {
struct MeshShape {};
struct IDevice { int id() const; };
class MeshDevice { public: std::vector<IDevice*> get_devices() const;
    CoreCoord compute_with_storage_grid_size() const; MeshShape shape() const;
    CoreCoord worker_core_from_logical_core(const CoreCoord&) const;
    uint32_t num_dram_channels() const;
    const std::unique_ptr<Allocator>& allocator() const; };
class MeshEvent {};
class MeshCommandQueue { public:
    MeshEvent enqueue_record_event(); MeshEvent enqueue_record_event_to_host();
    void enqueue_wait_for_event(const MeshEvent&); void finish();
    template <class B> void enqueue_read_mesh_buffer(void*, const B&, bool); };
struct MeshCoordinate { MeshCoordinate(uint32_t, uint32_t) {} };
struct MeshCoordinateRange { explicit MeshCoordinateRange(MeshShape) {}
    bool operator<(const MeshCoordinateRange&) const { return false; } };
struct MeshCoordinateRangeSet { explicit MeshCoordinateRangeSet(MeshCoordinateRange) {} };
struct DeviceLocalBufferConfig { uint64_t page_size = 0; BufferType buffer_type = BufferType::DRAM; };
struct ReplicatedBufferConfig { std::size_t size = 0; };
class MeshBuffer { public:
    static std::shared_ptr<MeshBuffer> create(const ReplicatedBufferConfig&, const DeviceLocalBufferConfig&, MeshDevice*, std::optional<uint64_t> address = std::nullopt);
    uint64_t address() const; std::size_t size() const;
};
struct MeshWorkload {
    void add_program(const MeshCoordinateRange&, Program&&);
    std::map<MeshCoordinateRange, Program>& get_programs();
};
template <class T> void EnqueueWriteMeshBuffer(MeshCommandQueue&, std::shared_ptr<MeshBuffer>, std::vector<T>&, bool = false);
template <class T> void EnqueueWriteMeshBuffer(MeshCommandQueue&, std::shared_ptr<MeshBuffer>, const std::vector<T>&, bool = false);
template <class T> void EnqueueReadMeshBuffer(MeshCommandQueue&, std::vector<T>&, std::shared_ptr<MeshBuffer>, bool);
void EnqueueMeshWorkload(MeshCommandQueue&, MeshWorkload&, bool);
void Finish(MeshCommandQueue&);
template <class T> void ReadShard(MeshCommandQueue&, std::vector<T>&, const std::shared_ptr<MeshBuffer>&, const MeshCoordinate&, bool = true);
void EventSynchronize(const MeshEvent&);
}  // namespace distributed
}  // namespace tt_metal
}  // namespace tt
