-- CreateEnum
CREATE TYPE "SlotType" AS ENUM ('WHEEL', 'NO_WHEEL');

-- AlterTable
ALTER TABLE "GroupEnrollment" ADD COLUMN     "preferredSlotType" "SlotType";

-- AlterTable
ALTER TABLE "SessionRegistration" ADD COLUMN     "slotType" "SlotType";
